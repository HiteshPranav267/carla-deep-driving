import carla
import cv2
import pygame
import numpy as np
import os
import csv
import time
import math
import random

# Import the Indian Chaos generation functions
from indian_traffic_manager import spawn_traffic, spawn_static_obstacles, spawn_pedestrians, manage_chaos

class AutomatedDataCollector:
    def __init__(self, target_frames_per_town=10000, towns=["Town03", "Town04"]):
        self.target_frames = target_frames_per_town
        self.towns = towns
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0) # High timeout for map loads
        
        # Dataset storage
        self.dataset_dir = "dataset"
        self.img_dir = os.path.join(self.dataset_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        
        self.csv_file_path = os.path.join(self.dataset_dir, "log.csv")
        self.csv_file = open(self.csv_file_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        
        # Check if empty
        if os.stat(self.csv_file_path).st_size == 0:
            self.csv_writer.writerow(["frame", "speed", "throttle", "steer", "brake", "town"])
            
        # UI
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((800, 600))
        self.font = pygame.font.SysFont("Arial", 24)
        
        # Global state
        self.total_frames_saved = 0
        self.collision_count = 0 
        self.surface = None
        self.image = None
        
    def run_collection(self):
        for town_name in self.towns:
            print(f"============== PREPARING FOR {town_name} ==============")
            self.collect_for_town(town_name)
            
        print("Data Collection Complete across all towns!")
        self.csv_file.close()
        pygame.quit()

    def collect_for_town(self, town_name):
        # 1. Load world with retry and extra safety
        max_retries = 3
        for i in range(max_retries):
            try:
                time.sleep(5.0) # Pre-load cooldown
                print(f"Attempting to load {town_name} (Trial {i+1})...")
                self.client.load_world(town_name)
                print("World loaded. Waiting for sync...")
                time.sleep(10.0) # Post-load sync
                self.world = self.client.get_world()
                self.tm = self.client.get_trafficmanager(8000)
                break
            except Exception as e:
                print(f"Load failed: {e}. Retrying in 5s...")
                time.sleep(5.0)
                if i == max_retries - 1: raise e
        
        # 2. Setup Physics & Sync Mode
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5) # Global safety increased
        self.tm.set_hybrid_physics_mode(False) 
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        print(f"Starting chaos generation in {town_name}...")
        
        vehicles, bikes = spawn_traffic(self.client, self.world, self.tm, 100)
        walkers, walker_controllers = spawn_pedestrians(self.client, self.world, 50)
        stuck_tracker = {}
        
        # 5. Spawn Ego Vehicle
        self.setup_ego_vehicle()
        self.setup_camera()
        self.setup_collision_sensor()
        
        # Wait for camera to send first frame
        print("Waiting for camera sensor...")
        while self.surface is None:
            self.world.tick()
            time.sleep(0.1)
            
        frames_in_town = 0
        clock = pygame.time.Clock()
        self.ego_stuck_tracker = {'time': time.time(), 'stage': 'normal', 'start': time.time()}
        
        print(f"Recording {self.target_frames} frames in {town_name}...")
        
        try:
            while frames_in_town < self.target_frames:
                self.world.tick()
                now = time.time()
                
                # Check UI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("User aborted.")
                        return

                # Manage Indian traffic behaviors
                manage_chaos(self.world, self.tm, vehicles, bikes, stuck_tracker)
                
                if self.surface is not None and self.image is not None:
                    # Rendering
                    self.display.blit(self.surface, (0, 0))
                    
                    v = self.ego_vehicle.get_velocity()
                    speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                    c = self.ego_vehicle.get_control()
                    t = self.ego_vehicle.get_transform()
                    rot = t.rotation
                    
                    # Traffic Light Logic
                    tl_state = self.ego_vehicle.get_traffic_light_state()
                    tl_str = str(tl_state).split('.')[-1] if self.ego_vehicle.is_at_traffic_light() else "None"
                    
                    # HUD Rendering
                    HUD_COLOR = (255, 255, 255)
                    info_texts = [
                        f"Map: {town_name}",
                        f"Frames: {frames_in_town}/{self.target_frames}",
                        f"Total Saved: {self.total_frames_saved}",
                        f"Speed: {speed:.1f} km/h",
                        f"Signal: {tl_str}",
                        f"--- Telemetry ---",
                        f"Pitch: {rot.pitch:.2f}°",
                        f"Roll: {rot.roll:.2f}°",
                        f"Yaw: {rot.yaw:.2f}°",
                        f"Collisions: {self.collision_count}",
                        f"--- Controls ---",
                        f"Throttle: {c.throttle:.2f}",
                        f"Steer: {c.steer:.2f}",
                        f"Brake: {c.brake:.2f}"
                    ]
                    
                    for i, text in enumerate(info_texts):
                        surface = self.font.render(text, True, HUD_COLOR)
                        self.display.blit(surface, (10, 10 + i * 25))
                        
                    pygame.display.flip()
                    
                    # Save Image
                    frame_name = f"{self.total_frames_saved:06d}.jpg"
                    frame_path = os.path.join(self.img_dir, frame_name)
                    cv2.imwrite(frame_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
                    
                    # Save Label
                    self.csv_writer.writerow([
                        frame_name, speed, c.throttle, c.steer, c.brake, town_name
                    ])
                    if frames_in_town % 1000 == 0:
                        print(f"Town {town_name}: {frames_in_town}/{self.target_frames} frames collected.")
                        self.csv_file.flush()
                        
                    frames_in_town += 1
                    self.total_frames_saved += 1
                    
                clock.tick(20) # 20 FPS target
                
        except KeyboardInterrupt:
            print("Skipping to next town or cleaning up...")
        except Exception as e:
            print(f"CRITICAL ERROR in {town_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"Cleaning up {town_name}...")
            # Clean up
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            self.camera.destroy()
            self.collision_sensor.destroy()
            self.ego_vehicle.destroy()
            
            self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in walkers])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in walker_controllers])
            
            time.sleep(2) # Give CARLA a moment before loading next town


    def setup_ego_vehicle(self):
        blueprint = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        self.ego_vehicle = None
        for sp in spawn_points:
            self.ego_vehicle = self.world.try_spawn_actor(blueprint, sp)
            if self.ego_vehicle:
                break
        
        if not self.ego_vehicle:
            raise RuntimeError("Could not spawn Ego vehicle at any spawn point!")
            
        # Ego autopilot is aggressive
        self.ego_vehicle.set_autopilot(True, self.tm.get_port())
        
        # Override safety for Ego to ensure it NEVER crashes for data quality
        self.tm.distance_to_leading_vehicle(self.ego_vehicle, 4.0)
        self.tm.ignore_vehicles_percentage(self.ego_vehicle, 0.0)
        self.tm.ignore_lights_percentage(self.ego_vehicle, 0.0)
        self.tm.vehicle_percentage_speed_difference(self.ego_vehicle, -20) # Drive slightly slower for better safety 

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
        self.image = array[:, :, :3]
        self.surface = pygame.surfarray.make_surface(self.image.swapaxes(0, 1))

    def setup_collision_sensor(self):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collision_count += 1
        print(f"Collision {self.collision_count} detected with {event.other_actor.type_id}")

if __name__ == "__main__":
    # You can specify exact towns and frames here
    # E.g. 10000 frames each for Town03 and Town04
    collector = AutomatedDataCollector(target_frames_per_town=10000, towns=["Town03", "Town04"])
    collector.run_collection()
