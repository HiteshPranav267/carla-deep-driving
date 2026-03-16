import carla
import cv2
import numpy as np
import os
import csv
import time
import random
import re
from PIL import Image
import torch

class CarlaDataCollector:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)

        # Portable path resolution: finds the 'dataset' folder relative to this script
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.normpath(os.path.join(base_path, 'dataset'))
        self.images_dir = os.path.normpath(os.path.join(self.dataset_dir, 'images'))
        
        os.makedirs(self.images_dir, exist_ok=True)

        # Weather Presets (name, preset) so logs are human-readable and verifiable.
        self.weather_presets = [
            ('ClearNoon', carla.WeatherParameters.ClearNoon),
            ('CloudyNoon', carla.WeatherParameters.CloudyNoon),
            ('WetNoon', carla.WeatherParameters.WetNoon),
            ('HardRainNoon', carla.WeatherParameters.HardRainNoon),
            ('SoftRainNoon', carla.WeatherParameters.SoftRainNoon),
            ('ClearSunset', carla.WeatherParameters.ClearSunset),
            ('WetCloudyNoon', carla.WeatherParameters.WetCloudyNoon)
        ]

        self.ego_blueprints = ['vehicle.tesla.model3', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer', 'vehicle.toyota.prius']
        # Safe, stable vehicles for background to prevent physics explosions
        self.bg_blueprints = ['vehicle.tesla.model3', 'vehicle.audi.tt', 'vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.citroen.c3', 'vehicle.nissan.micra']
        self.training_towns = ['Town01', 'Town02', 'Town04']
        self.frames_per_town = 20000
        
        # CSV logging
        self.csv_file_path = os.path.join(self.dataset_dir, 'log.csv')
        print(f"Dataset location: {self.dataset_dir}")
        self.collected_by_town = {town: 0 for town in self.training_towns}
        self.last_weather_name = None
        self.resume_start_town = os.environ.get('START_TOWN', '').strip()
        
        file_exists = os.path.isfile(self.csv_file_path)
        self.csv_file = open(self.csv_file_path, 'a', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists or os.path.getsize(self.csv_file_path) == 0:
            self.csv_writer.writerow([
                'timestamp', 'frame_id', 'speed', 'speed_limit', 'throttle', 'brake', 'steer', 
                'gap', 'status', 'lane_id', 'road_id', 'weather', 'town'
            ])
            self.csv_file.flush()

        self.total_frames_collected = self._initialize_resume_state()
        if self.resume_start_town:
            print(f"Resume override enabled via START_TOWN={self.resume_start_town}")
        print(f"Resuming frame index from: {self.total_frames_collected}")
        print(f"Existing town progress: {self.collected_by_town}")
        self.session_id = 0
        self.collision_event = False
        self.recovery_mode = False
        self.recovery_timer = 0
        self.current_frame = None
        self.vis_frame = None
        self.ego_vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.bg_actors = []
        self.stuck_ticks = 0

    def _initialize_resume_state(self):
        """Restore frame index and per-town counts from existing CSV so runs can resume safely."""
        if not os.path.isfile(self.csv_file_path) or os.path.getsize(self.csv_file_path) == 0:
            return 0

        max_frame_index = -1
        try:
            with open(self.csv_file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    town = (row.get('town') or '').strip()
                    if town in self.collected_by_town:
                        self.collected_by_town[town] += 1

                    frame_id = (row.get('frame_id') or '').strip()
                    match = re.match(r'^(\d+)\.(jpg|png)$', frame_id, re.IGNORECASE)
                    if match:
                        frame_num = int(match.group(1))
                        if frame_num > max_frame_index:
                            max_frame_index = frame_num
        except Exception as e:
            print(f"Warning: Could not parse existing CSV for resume state ({e}).")

        return max_frame_index + 1

    def _choose_weather(self, exclude_name=None):
        """Pick weather and avoid repeating the previous town's weather when possible."""
        candidates = self.weather_presets
        if exclude_name and len(self.weather_presets) > 1:
            filtered = [w for w in self.weather_presets if w[0] != exclude_name]
            if filtered:
                candidates = filtered
        weather_name, weather = random.choice(candidates)
        return weather_name, weather

    def collision_callback(self, event):
        self.collision_event = True

    def configure_traffic_manager(self, tm):
        """Globally configure Traffic Manager for safer driving."""
        tm.set_global_distance_to_leading_vehicle(4.0)
        # Disable hybrid mode to ensure all NPCs move reliably with physics
        tm.set_hybrid_physics_mode(False) 

    def configure_vehicle_tm(self, tm, vehicle, is_ego=False):
        """Per-vehicle TM tuning."""
        if is_ego:
            tm.auto_lane_change(vehicle, True)
            tm.distance_to_leading_vehicle(vehicle, 2.0)
            tm.vehicle_percentage_speed_difference(vehicle, -10.0) 
        else:
            # Keep NPC traffic calm and predictable for stable dataset generation.
            tm.auto_lane_change(vehicle, False)
            tm.distance_to_leading_vehicle(vehicle, 5.0)
            # Keep NPCs slightly slower than the speed limit to reduce chaotic interactions.
            tm.vehicle_percentage_speed_difference(vehicle, random.uniform(8.0, 20.0))
            # Do not run red lights; this reduces random collisions/spins at junctions.
            tm.ignore_lights_percentage(vehicle, 0)
            # Disable random lane-change impulses when API is available.
            try:
                tm.random_left_lanechange_percentage(vehicle, 0)
                tm.random_right_lanechange_percentage(vehicle, 0)
            except Exception:
                pass

    def check_pileup(self, speed, nearby):
        """Strategic Respawn: If we are stuck in a pileup, get out of there."""
        
        # --- ROBUST CHECK ---
        # 1. Don't count as stuck if we are waiting for a RED traffic light
        if nearby['tl_state'] == 0: # Red
             self.stuck_ticks = 0
             return False

        # 2. If speed is near zero and there are blocked vehicles ahead, or we've been stuck
        if (nearby['blocked_ahead'] >= 1 and speed < 2.0) or (speed < 0.2):
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        # 3. Increase threshold: ~40 seconds of being 'stuck' at 10 FPS (10 Hz)
        if self.stuck_ticks > 400:  
            print("  Stuck/Pileup detected. Performing Strategic Respawn to clear road...")
            return True
        return False

    def get_status(self, speed, control, nearby):
        if self.collision_event:
            self.collision_event = False
            self.recovery_mode = True
            self.recovery_timer = 30 
            return "Emergency Stop"
        
        if self.recovery_mode:
            self.recovery_timer -= 1
            if self.recovery_timer <= 0:
                self.recovery_mode = False
                return "Recovered"
            return "Caution"

        if control.gear < 0:
            return "Reverse"
        
        if nearby['tl_state'] == 0: # Red
            if nearby['is_at_tl'] or nearby['dist_to_tl'] < 15.0:
                return "Red Light"
            if nearby['dist_to_lead'] < 12.0 and nearby['lead_v_speed'] < 2.0 and nearby['dist_to_tl'] < 35.0:
                return "Red Light"
        
        if nearby['tl_state'] == 1: # Yellow
            if nearby['is_at_tl'] or nearby['dist_to_tl'] < 15.0:
                return "Yellow Light"
            if nearby['dist_to_lead'] < 12.0 and nearby['lead_v_speed'] < 5.0 and nearby['dist_to_tl'] < 35.0:
                 return "Yellow Light"

        if nearby['dist_to_lead'] < 8.0:
            return "Caution"

        if speed < 0.5 and control.brake > 0.5:
            return "Emergency Stop"

        return "Following Route"

    def get_nearby_info(self, world, ego_vehicle):
        ego_location = ego_vehicle.get_location()
        ego_waypoint = world.get_map().get_waypoint(ego_location)
        ego_transform = ego_vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()
        
        dist_to_lead = 100.0
        lead_v_speed = 100.0
        blocked_ahead = 0
        
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.id == ego_vehicle.id: continue
            v_loc = vehicle.get_location()
            dist = ego_location.distance(v_loc)
            if dist < 50.0:
                v_waypoint = world.get_map().get_waypoint(v_loc)
                if v_waypoint.road_id == ego_waypoint.road_id and v_waypoint.lane_id == ego_waypoint.lane_id:
                    ray = v_loc - ego_location
                    if (ego_forward.x * ray.x + ego_forward.y * ray.y) > 0:
                        v_vel = vehicle.get_velocity()
                        v_speed = np.sqrt(v_vel.x**2 + v_vel.y**2 + v_vel.z**2) * 3.6
                        if dist < dist_to_lead:
                            dist_to_lead = dist
                            lead_v_speed = v_speed
                        if v_speed < 1.0 and dist < 20.0:
                            blocked_ahead += 1

        tl_state = 2 # Green
        is_at_tl = ego_vehicle.is_at_traffic_light()
        dist_to_tl = 100.0
        
        active_tl = ego_vehicle.get_traffic_light()
        if active_tl:
            tl_state = int(active_tl.get_state())
            dist_to_tl = ego_location.distance(active_tl.get_location())
        else:
            traffic_lights = world.get_actors().filter('traffic.traffic_light')
            for tl in traffic_lights:
                tl_loc = tl.get_location()
                d = ego_location.distance(tl_loc)
                if d < 40.0:
                    ray = tl_loc - ego_location
                    if (ego_forward.x * ray.x + ego_forward.y * ray.y) > 0:
                        mag = np.sqrt(ray.x**2 + ray.y**2) * np.sqrt(ego_forward.x**2 + ego_forward.y**2)
                        dot = (ego_forward.x * ray.x + ego_forward.y * ray.y)
                        if dot / (mag + 1e-6) > 0.7 and d < dist_to_tl: 
                            dist_to_tl = d
                            tl_state = int(tl.get_state())

        return {
            'lane_id': ego_waypoint.lane_id, 'road_id': ego_waypoint.road_id,
            'tl_state': tl_state, 'is_at_tl': is_at_tl,
            'dist_to_lead': dist_to_lead, 'lead_v_speed': lead_v_speed,
            'dist_to_tl': dist_to_tl, 'blocked_ahead': blocked_ahead,
            'speed_limit': ego_vehicle.get_speed_limit()
        }

    def collect_data(self):
        print("Starting Data Collection (Sequential Towns)...")
        
        # --- ROBUST RESET ---
        try:
            self.client.set_timeout(5.0) 
            world = self.client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass

        self.client.set_timeout(120.0) 

        towns_to_run = list(self.training_towns)
        if self.resume_start_town and self.resume_start_town in towns_to_run:
            start_idx = towns_to_run.index(self.resume_start_town)
            towns_to_run = towns_to_run[start_idx:]

        for town_name in towns_to_run:
            print(f"\n--- Preparing {town_name} ---")
            frames_in_town = self.collected_by_town.get(town_name, 0)
            if frames_in_town >= self.frames_per_town:
                print(f"{town_name} already complete ({frames_in_town}/{self.frames_per_town}). Skipping.")
                continue
            print(f"Resuming {town_name}: {frames_in_town}/{self.frames_per_town} frames already collected.")
            
            # --- FAULT-TOLERANT MAP LOADING ---
            try:
                world = self.client.get_world()
                current_map_name = world.get_map().name
                
                if town_name not in current_map_name:
                    print(f"Attempting to switch map to {town_name}...")
                    try:
                        self.client.load_world(town_name)
                    except RuntimeError as e:
                        print(f"(!) Warning: Could not load {town_name} automatically ({e}).")
                        print("    Continuing with the CURRENT map. Data will be saved under the requested town name.")
                else:
                    print(f"Already on {town_name}. Skipping load.")
                    
            except Exception as e:
                print(f"Critical Simulator Error: {e}")
                continue

            # --- SETUP SESSION ---
            try:
                time.sleep(2.0) 
                world = self.client.get_world()
                
                # 1. Set world to synchronous mode first
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.1
                world.apply_settings(settings)

                # 2. Then configure Traffic Manager to match
                tm = self.client.get_trafficmanager(8000)
                tm.set_synchronous_mode(True)
                self.configure_traffic_manager(tm)

                while frames_in_town < self.frames_per_town:
                    weather_name, weather = self._choose_weather(exclude_name=self.last_weather_name)
                    self.last_weather_name = weather_name
                    world.set_weather(weather)
                    print(f"  Weather set to: {weather_name}")
                    
                    self.spawn_ego_vehicle(world, tm)
                    if self.ego_vehicle is None: 
                        # If spawn failed, try one more time then skip
                        print("Retrying spawn...")
                        continue
                        
                    self.spawn_background_traffic(world, tm)
                    print(f"  Spawned {len(self.bg_actors)} background vehicles.")
                    
                    collected = self.collect_in_world(
                        world, town_name, weather_name, self.frames_per_town - frames_in_town
                    )
                    frames_in_town += collected
                    self.collected_by_town[town_name] = frames_in_town
                    self.cleanup_session()
                    if self.total_frames_collected > 1000000: break 
            except Exception as e:
                print(f"Session Error: {e}")
                self.cleanup_session()
                continue
        
        self.csv_file.close()
        cv2.destroyAllWindows()

    def collect_in_world(self, world, town_name, weather_name, max_to_collect):
        frames_collected = 0
        spectator = world.get_spectator()
        self.stuck_ticks = 0
        if self.ego_vehicle is None:
            return 0
        last_loc = self.ego_vehicle.get_location()

        while frames_collected < max_to_collect:
            world.tick()
            if not self.ego_vehicle or not self.ego_vehicle.is_alive: break

            # Spectator
            transform = self.ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + transform.get_forward_vector() * -10 + carla.Location(z=5),
                carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
            ))

            # Data Extraction
            nearby = self.get_nearby_info(world, self.ego_vehicle)
            velocity = self.ego_vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            control = self.ego_vehicle.get_control()
            timestamp = world.get_snapshot().timestamp.elapsed_seconds
            
            # Status Logic
            status = self.get_status(speed, control, nearby)

            # Strategic Respawn Check
            if self.check_pileup(speed, nearby):
                break

            # Visualization (HUD)
            if self.vis_frame is not None:
                cv_image = cv2.cvtColor(self.vis_frame, cv2.COLOR_RGB2BGR)
                overlay = cv_image.copy()
                cv2.rectangle(overlay, (0, 0), (280, 270), (0, 0, 0), -1)
                cv_image = cv2.addWeighted(overlay, 0.6, cv_image, 0.4, 0)
                y_pos = [30]
                def draw_stat(label, val, color=(255, 255, 255)):
                    cv2.putText(cv_image, f"{label}: {val}", (15, y_pos[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos[0] += 25
                draw_stat("TOWN", town_name)
                status_color = (0, 255, 0)
                if status == "Emergency Stop": status_color = (0, 0, 255)
                if status == "Red Light": status_color = (0, 0, 255)
                if status == "Yellow Light": status_color = (0, 255, 255)
                if status == "Caution": status_color = (0, 165, 255)
                draw_stat("STATUS", status, status_color)
                draw_stat("SPEED", f"{speed:.1f} / {nearby['speed_limit']:.0f} km/h")
                draw_stat("GAP", f"{nearby['dist_to_lead']:.1f} m")
                draw_stat("BLOCKED", f"{nearby['blocked_ahead']}")
                draw_stat("TOTAL", f"{self.total_frames_collected}")
                cv2.imshow("CARLA Research Data Collector", cv_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.total_frames_collected = 10000000
                    return frames_collected

            # Save Data
            if self.current_frame is not None:
                frame_name = f"{self.total_frames_collected:06d}.jpg"
                self.current_frame.save(os.path.join(self.images_dir, frame_name))
                self.csv_writer.writerow([
                    timestamp, frame_name, speed, nearby['speed_limit'], control.throttle, control.brake, control.steer,
                    nearby['dist_to_lead'], status, nearby['lane_id'], nearby['road_id'], weather_name, town_name
                ])
                self.csv_file.flush()
                self.total_frames_collected += 1
                frames_collected += 1

        return frames_collected

    def spawn_ego_vehicle(self, world, tm):
        bp = world.get_blueprint_library().find(random.choice(self.ego_blueprints))
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for sp in spawn_points:
            if self._is_spawn_clear(world, sp.location):
                self.ego_vehicle = world.try_spawn_actor(bp, sp)
                if self.ego_vehicle: break
        if self.ego_vehicle is None: return
        self.ego_vehicle.set_autopilot(True, tm.get_port())
        self.configure_vehicle_tm(tm, self.ego_vehicle, is_ego=True)
        self.setup_camera(world)
        self.collision_sensor = world.spawn_actor(
            world.get_blueprint_library().find('sensor.other.collision'),
            carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(lambda event: self.collision_callback(event))

    def _is_spawn_clear(self, world, location):
        actors = world.get_actors().filter('vehicle.*')
        for actor in actors:
            if actor.get_location().distance(location) < 10.0:
                return False
        return True

    def spawn_background_traffic(self, world, tm):
        self.bg_actors = []
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        spawned = 0
        for sp in spawn_points:
            if spawned >= 40: break 
            if self._is_spawn_clear(world, sp.location):
                bp = world.get_blueprint_library().find(random.choice(self.bg_blueprints))
                actor = world.try_spawn_actor(bp, sp)
                if actor:
                    actor.set_autopilot(True, tm.get_port())
                    self.configure_vehicle_tm(tm, actor, is_ego=False)
                    self.bg_actors.append(actor)
                    spawned += 1

    def setup_camera(self, world):
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '640'); bp.set_attribute('image_size_y', '480')
        self.camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=self.ego_vehicle)
        self.camera.listen(self.process_image)
        self.current_frame = None; self.vis_frame = None

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.vis_frame = array.copy()
        self.current_frame = Image.fromarray(cv2.resize(array, (224, 224)))

    def cleanup_session(self):
        if hasattr(self, 'camera') and self.camera: self.camera.destroy()
        if hasattr(self, 'collision_sensor') and self.collision_sensor: self.collision_sensor.destroy()
        if hasattr(self, 'ego_vehicle') and self.ego_vehicle: self.ego_vehicle.destroy()
        if hasattr(self, 'bg_actors'):
            for a in self.bg_actors: 
                if a.is_alive: a.destroy()

if __name__ == "__main__":
    CarlaDataCollector().collect_data()
