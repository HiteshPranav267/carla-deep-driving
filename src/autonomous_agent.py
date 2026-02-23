import carla
import cv2
import pygame
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import EndToEndDrivingModel

class AutonomousAgent:
    def __init__(self, model_path="best_model.pth"):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((800, 600))
        self.font = pygame.font.SysFont("Arial", 24)
        
        # Carla Setup
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Load Deep Learning Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading End-To-End model on {self.device}...")
        
        self.model = EndToEndDrivingModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print("WARNING: best_model.pth not found. The agent will drive randomly!")
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_array = None
        self.setup_ego_vehicle()
        self.setup_camera()
        
    def setup_ego_vehicle(self):
        blueprint = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        spawn_point = self.world.get_map().get_spawn_points()[10] # pick random
        
        self.ego_vehicle = self.world.spawn_actor(blueprint, spawn_point)
        # NO AUTOPILOT. WE DRIVE THIS OURSELVES LITERALLY END-TO-END.
        print("Spawned Autonomous Test Ego Vehicle. AI Taking Control.")

    def setup_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "800")
        bp.set_attribute("image_size_y", "600")
        bp.set_attribute("fov", "90")
        
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(bp, transform, attach_to=self.ego_vehicle)
        self.camera.listen(self.process_image)

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.image_array = array[:, :, :3]
        
    def run(self):
        clock = pygame.time.Clock()
        
        try:
            while True:
                self.world.tick()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                if self.image_array is not None:
                    # Rendering
                    surface = pygame.surfarray.make_surface(self.image_array.swapaxes(0, 1))
                    self.display.blit(surface, (0, 0))
                    
                    # 1. Prepare inputs for Neural Network
                    img_pil = Image.fromarray(cv2.cvtColor(self.image_array, cv2.COLOR_RGB2BGR))
                    img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                    
                    v = self.ego_vehicle.get_velocity()
                    speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                    speed_tensor = torch.tensor([[speed]], dtype=torch.float32).to(self.device)
                    
                    # 2. Run DL inference
                    with torch.no_grad():
                        action = self.model(img_tensor, speed_tensor)[0]
                    
                    throttle = action[0].item()
                    steer = action[1].item()
                    brake = action[2].item()
                    
                    # 3. Apply explicit learned control back to environment (NO RULES)
                    control = carla.VehicleControl()
                    control.throttle = throttle
                    control.steer = steer
                    control.brake = brake
                    self.ego_vehicle.apply_control(control)
                    
                    # Dashboard HUD
                    text_title = self.font.render("FULLY NEURAL MODEL DRIVING", True, (0, 255, 0))
                    text_speed = self.font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255))
                    text_steer = self.font.render(f"Net Steer: {steer:.2f}", True, (255, 255, 255))
                    text_throttle = self.font.render(f"Net Throttle: {throttle:.2f}", True, (255, 255, 255))
                    text_brake = self.font.render(f"Net Brake: {brake:.2f}", True, (255, 255, 255))
                    
                    self.display.blit(text_title, (10, 10))
                    self.display.blit(text_speed, (10, 40))
                    self.display.blit(text_steer, (10, 80))
                    self.display.blit(text_throttle, (10, 110))
                    self.display.blit(text_brake, (10, 140))
                    
                    pygame.display.flip()

                clock.tick(20)

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.destroy()
            self.ego_vehicle.destroy()
            pygame.quit()

if __name__ == "__main__":
    agent = AutonomousAgent()
    agent.run()
