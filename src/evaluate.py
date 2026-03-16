import carla
import torch
import numpy as np
import pandas as pd
import os
import time
import glob
import random
from collections import deque
from torchvision import transforms
from model import create_model
from PIL import Image
import cv2

class CarlaEvaluator:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # Evaluation towns (Town03, Town05 only)
        self.evaluation_towns = ['Town03', 'Town05']

        # Weather conditions
        self.weather_conditions = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.WetNoon  # Rain + fog combined
        ]

        # Model names to evaluate. Checkpoints are resolved dynamically.
        self.models = ['baseline_cnn', 'cnn_lstm']

        # Results storage
        self.results = []
        self.results_file = '../results/evaluation_log.csv'
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def evaluate(self):
        print("Starting evaluation...")
        print("Evaluating on Town03 and Town05 only")
        print("Weather conditions: Clear and Adverse (Rain + Fog)")

        # Create results directory
        os.makedirs('../results', exist_ok=True)

        # Load models
        models = {}
        for model_name in self.models:
            model = self.load_model_for_evaluation(model_name)
            if model is not None:
                models[model_name] = model

        if not models:
            print("No models found to evaluate. Please train models first.")
            return

        # Evaluation loop
        for town_idx, town_name in enumerate(self.evaluation_towns):
            for weather_idx, weather in enumerate(self.weather_conditions):
                for model_name, model in models.items():
                    for episode in range(10):  # 10 episodes per town/weather/model
                        print(f"\nEvaluating {model_name} in {town_name} | Weather {weather_idx+1}/2 | Episode {episode+1}/10")
                        results = self.run_episode(town_name, weather, model, model_name, episode, town_idx, weather_idx)
                        self.results.append(results)

        # Save results
        self.save_results()
        print(f"\nEvaluation complete! Results saved to {self.results_file}")

    def run_episode(self, town_name, weather, model, model_name, episode, town_idx, weather_idx):
        # Load town
        self.client.load_world(town_name)
        world = self.client.get_world()
        world.set_weather(weather)

        # Setup traffic manager
        tm = self.client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        # Setup synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Spawn ego vehicle
        vehicle = self.spawn_ego_vehicle(world, tm)
        if not vehicle:
            return self.create_empty_results(town_name, weather, model_name, episode, town_idx, weather_idx)

        # Setup camera sensor
        camera = self.setup_camera(world, vehicle)
        if not camera:
            return self.create_empty_results(town_name, weather, model_name, episode, town_idx, weather_idx)

        # Initialize variables
        start_time = time.time()
        route_completion = 0.0
        collisions = 0
        phantom_brake_events = 0
        lane_invasions = 0
        total_distance = 0.0
        speed_sum = 0.0
        frame_count = 0
        frame_buffer = deque(maxlen=5)

        # Get initial transform for route planning
        start_transform = vehicle.get_transform()

        try:
            while True:
                world.tick()

                # Get vehicle state
                if vehicle.is_alive:
                    velocity = vehicle.get_velocity()
                    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h
                    control = vehicle.get_control()

                    # Process camera image
                    if hasattr(self, 'current_frame') and self.current_frame is not None:
                        # Preprocess image consistently with training normalization.
                        frame_tensor = self.eval_transform(self.current_frame)  # [3, 224, 224]
                        frame_buffer.append(frame_tensor)

                        # CNN-LSTM needs a full temporal window of 5 frames.
                        if model_name == 'cnn_lstm' and len(frame_buffer) < 5:
                            continue

                        # Get model prediction
                        with torch.no_grad():
                            speed_normalized = torch.tensor([[speed / 100.0]], dtype=torch.float32)
                            if model_name == 'cnn_lstm':
                                image_tensor = torch.stack(list(frame_buffer), dim=0).unsqueeze(0)  # [1, 5, 3, 224, 224]
                            else:
                                image_tensor = frame_buffer[-1].unsqueeze(0)  # [1, 3, 224, 224]

                            prediction = model(image_tensor, speed_normalized).squeeze().numpy()

                            # Apply model controls (smoothed)
                            control.throttle = np.clip(prediction[0], 0, 1)
                            control.steer = np.clip(prediction[1], -1, 1)
                            control.brake = np.clip(prediction[2], 0, 1)

                            vehicle.apply_control(control)

                    # Check for collisions
                    if hasattr(self, 'collision_counter'):
                        if self.collision_counter > 0:
                            collisions += self.collision_counter
                            self.collision_counter = 0

                    # Check for phantom braking (Unit 2: noise robustness testing)
                    if control.brake > 0.5 and speed < 5 and not self.is_obstacle_near(world, vehicle):
                        phantom_brake_events += 1

                    # Update metrics
                    frame_count += 1
                    speed_sum += speed
                    total_distance += velocity.x * 0.05  # Approximate distance (delta_seconds = 0.05)

                    # Simple route completion check (distance-based)
                    current_transform = vehicle.get_transform()
                    distance_traveled = np.sqrt(
                        (current_transform.location.x - start_transform.location.x)**2 +
                        (current_transform.location.y - start_transform.location.y)**2
                    )
                    route_completion = min(distance_traveled / 1000.0, 1.0)  # Normalize to 0-1

                else:
                    break  # Vehicle destroyed

        except Exception as e:
            print(f"Episode exception: {e}")
        finally:
            # Cleanup
            if hasattr(self, 'camera') and self.camera and self.camera.is_alive:
                self.camera.destroy()
            if vehicle and vehicle.is_alive:
                vehicle.destroy()

            # Disable synchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)

        # Calculate final metrics
        episode_time = time.time() - start_time
        avg_speed = speed_sum / frame_count if frame_count > 0 else 0
        success = collisions == 0 and route_completion >= 0.8

        # Create results dictionary
        results = {
            'model': model_name,
            'town': town_name,
            'weather': 'clear' if weather_idx == 0 else 'adverse',
            'episode': episode,
            'route_completion': route_completion,
            'collisions': collisions,
            'phantom_brake_events': phantom_brake_events,
            'lane_invasions': lane_invasions,
            'average_speed_kmh': avg_speed,
            'episode_time_sec': episode_time,
            'total_distance_km': total_distance / 1000.0,
            'success': success
        }

        print(f"Results: {results}")
        return results

    def load_model_for_evaluation(self, model_name):
        """Load either an ensemble of member checkpoints or a legacy single checkpoint."""
        model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        member_pattern = os.path.join(model_dir, f"{model_name}_member_*.pth")
        member_paths = sorted(glob.glob(member_pattern))

        if member_paths:
            model = create_model(model_name, n_estimators=len(member_paths))
            for idx, path in enumerate(member_paths):
                state_dict = torch.load(path, map_location='cpu')
                model.models[idx].load_state_dict(state_dict)
            model.eval()
            print(f"Loaded {model_name} ensemble with {len(member_paths)} members from {model_dir}")
            return model

        legacy_path = os.path.join(model_dir, f"{model_name}.pth")
        if os.path.exists(legacy_path):
            model = create_model(model_name)
            model.load_state_dict(torch.load(legacy_path, map_location='cpu'))
            model.eval()
            print(f"Loaded legacy {model_name} checkpoint from {legacy_path}")
            return model

        print(
            f"Warning: No checkpoints found for {model_name}. "
            f"Looked for {model_name}_member_*.pth and {model_name}.pth in {model_dir}"
        )
        return None

    def spawn_ego_vehicle(self, world, tm):
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        # Find a good spawn point
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        ego_spawn_point = spawn_points[0]

        vehicle = world.try_spawn_actor(vehicle_bp, ego_spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, tm.get_port())
            tm.distance_to_leading_vehicle(vehicle, 3.0)
            tm.ignore_lights_percentage(vehicle, 0.0)
            tm.ignore_signs_percentage(vehicle, 0.0)
            tm.ignore_vehicles_percentage(vehicle, 0.0)

        return vehicle

    def setup_camera(self, world, vehicle):
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '224')
        bp.set_attribute('image_size_y', '224')
        bp.set_attribute('fov', '90')

        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(bp, transform, attach_to=vehicle)

        if camera:
            camera.listen(self.process_image)
        return camera

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb_image = array[:, :, :3]
        self.current_frame = Image.fromarray(rgb_image)

    def is_obstacle_near(self, world, vehicle):
        # Simple obstacle detection within 15m
        location = vehicle.get_location()
        map = world.get_map()

        # Check for nearby vehicles
        actors = world.get_actors()
        for actor in actors:
            if actor.type_id.startswith('vehicle.'):
                other_location = actor.get_location()
                distance = location.distance(other_location)
                if distance < 15:
                    return True
        return False

    def create_empty_results(self, town_name, weather, model_name, episode, town_idx, weather_idx):
        return {
            'model': model_name,
            'town': town_name,
            'weather': 'clear' if weather_idx == 0 else 'adverse',
            'episode': episode,
            'route_completion': 0.0,
            'collisions': 0,
            'phantom_brake_events': 0,
            'lane_invasions': 0,
            'average_speed_kmh': 0.0,
            'episode_time_sec': 0.0,
            'total_distance_km': 0.0,
            'success': False
        }

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        print(f"Saved {len(self.results)} evaluation results to {self.results_file}")

if __name__ == "__main__":
    evaluator = CarlaEvaluator()
    evaluator.evaluate()