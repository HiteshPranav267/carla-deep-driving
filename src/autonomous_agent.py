import carla
import cv2
import pygame
import numpy as np
import torch
import time
import os
import csv
from torchvision import transforms
from PIL import Image
from model import EndToEndDrivingModel

class HybridPilot:
    """
    IRON WALL VERSION (v2.4 - Silky Smooth + Logging):
    - Logging: Resets logs.csv every run to track telemetry.
    - Control Filter: Low-pass filter for Throttle/Brake to stop jitter.
    - Radar Buffer: Reduces phantom braking by filtering noise.
    """
    def __init__(self, model_path="models/best_model.pth"):
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Neural Iron Wall v2.4")
        self.font = pygame.font.SysFont("Arial", 18, bold=True)
        
        # Initialize Logs
        self.log_path = "logs.csv"
        self.log_file = open(self.log_path, mode='w', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["Timestamp", "Speed_KMH", "Throttle", "Brake", "Steer", "Gap", "Status"])
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        
        # Speed Management
        self.target_speed = 25.0 # Max Speed km/h
        self.smoothing_alpha = 0.2 # 0 to 1 (Lower = Smoother, Higher = Sharper)
        self.s_throttle = 0.0
        self.s_brake = 0.0
        
        # Connect to Traffic Manager
        self.tm = self.client.get_trafficmanager(8000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EndToEndDrivingModel().to(self.device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print("Model Ready.")
            except: pass
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.vehicle = None
        self.camera = None
        self.obstacle_sensor = None
        self.current_frame = None
        self.last_obstacle_dist = 50.0 # Start clear
        self.is_launching = False 
        
        self.setup_actors()

    def setup_actors(self):
        bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        spawn_point = np.random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        self.vehicle.set_autopilot(True)
        
        # TM Settings for ultra-smooth navigation
        self.tm.vehicle_percentage_speed_difference(self.vehicle, 60) # Smooth 40% speed limit
        self.tm.set_global_distance_to_leading_vehicle(4.0)
        
        # Camera
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: self.on_image(data))

        # Precision Radar (Raised for floor-filtering)
        obs_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
        obs_bp.set_attribute('distance', '15')
        obs_bp.set_attribute('hit_radius', '0.4')
        self.obstacle_sensor = self.world.spawn_actor(obs_bp, carla.Transform(carla.Location(x=2.2, z=1.0)), attach_to=self.vehicle)
        self.obstacle_sensor.listen(lambda data: self.update_obstacle(data))

    def update_obstacle(self, data):
        # Filter: Only accept the new distance if it's close, otherwise grow slowly
        if data.distance < self.last_obstacle_dist:
             self.last_obstacle_dist = data.distance # React fast to danger
        else:
             self.last_obstacle_dist = self.last_obstacle_dist * 0.9 + data.distance * 0.1 # Grow slow

    def on_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.current_frame = array[:, :, :3]

    def run_inference(self, speed_kmh):
        if self.current_frame is None: return 0.0, 0.0
        img = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        speed_t = torch.tensor([[speed_kmh/100.0]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(img_t, speed_t)[0]
        return float(output[0]), float(output[2])

    def main_loop(self):
        clock = pygame.time.Clock()
        try:
            while True:
                self.world.wait_for_tick()
                v = self.vehicle.get_velocity()
                speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                ctrl = self.vehicle.get_control()
                
                # 1. AI Decision
                raw_throttle, raw_brake = self.run_inference(speed_kmh)
                
                # 2. Smooth Control Filtering (Shock Absorber)
                self.s_throttle = (self.smoothing_alpha * raw_throttle) + ((1 - self.smoothing_alpha) * self.s_throttle)
                self.s_brake = (self.smoothing_alpha * raw_brake) + ((1 - self.smoothing_alpha) * self.s_brake)
                
                target_throttle = self.s_throttle
                target_brake = self.s_brake
                
                # 3. Dynamic Safety State
                safety_s = "CLEAR"

                # A. Proximity (Hard Stops)
                if self.last_obstacle_dist < 4.0:
                    target_throttle, target_brake = 0.0, 1.0
                    safety_s = "STOPPING"
                    self.is_launching = False
                elif self.last_obstacle_dist < 10.0:
                    target_throttle, target_brake = 0.0, 0.4
                    safety_s = "GAP HOLD"
                    self.is_launching = False
                
                # B. Turn Smoothing (If steering, cap the max gas)
                elif abs(ctrl.steer) > 0.12:
                    target_throttle = min(target_throttle, 0.18)
                    safety_s = "TURN SMOOTHING"

                # C. Launch Assist (Fixing 0kmh stickiness)
                else:
                    if speed_kmh < 1.0: self.is_launching = True
                    if speed_kmh > 15.0: self.is_launching = False
                    
                    if self.is_launching:
                        target_throttle = max(target_throttle, 0.45)
                        target_brake = 0.0
                        safety_s = "LAUNCHING"

                # Final Limiters
                if speed_kmh > self.target_speed: target_throttle = 0.0

                # Apply
                ctrl.throttle = target_throttle
                ctrl.brake = target_brake
                self.vehicle.apply_control(ctrl)

                # Decay Radar
                self.last_obstacle_dist = min(50.0, self.last_obstacle_dist + 0.5)

                # 4. Telemetry Logging
                self.log_writer.writerow([
                    time.time(), 
                    round(speed_kmh, 2), 
                    round(target_throttle, 2), 
                    round(target_brake, 2), 
                    round(ctrl.steer, 2), 
                    round(self.last_obstacle_dist, 1), 
                    safety_s
                ])

                self.render(speed_kmh, target_throttle, target_brake, safety_s, ctrl.steer)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return

                clock.tick(20)

        finally:
            print("Cleaning up...")
            try:
                if hasattr(self, 'log_file'): self.log_file.close()
                if self.camera: self.camera.destroy()
                if self.obstacle_sensor: self.obstacle_sensor.destroy()
                if self.vehicle: self.vehicle.destroy()
            except: pass
            pygame.quit()

    def render(self, speed, thr, brk, status, steer):
        if self.current_frame is not None:
            rgb_img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            overlay = pygame.Surface((320, 200))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.display.blit(overlay, (10, 10))
            
            stats = [
                f"IRON WALL v2.4 (SMOOTH)",
                f"Status: {status}",
                f"Speed: {speed:.1f} km/h",
                f"Gap: {self.last_obstacle_dist:.1f}m",
                f"Throttle: {thr:.2f}",
                f"Brake: {brk:.2f}",
                f"Steer: {steer:.2f}"
            ]
            for i, text in enumerate(stats):
                color = (0, 255, 100) if "CLEAR" in text or "v2.4" in text else (255, 255, 255)
                if "STOP" in text or "GAP" in text: color = (255, 50, 50)
                surf = self.font.render(text, True, color)
                self.display.blit(surf, (20, 20 + i*22))
        pygame.display.flip()

if __name__ == "__main__":
    pilot = HybridPilot()
    pilot.main_loop()
