import carla
import torch
import numpy as np
import pandas as pd
import os
import time
from model import create_model

class ScenarioTester:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # Models to test
        self.models = {
            'baseline_cnn': '../models/baseline_cnn.pth',
            'cnn_lstm': '../models/cnn_lstm.pth'
        }

        # Load models
        self.loaded_models = {}
        for model_name, model_path in self.models.items():
            if os.path.exists(model_path):
                model = create_model(model_name)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                self.loaded_models[model_name] = model
                print(f"Loaded {model_name} for scenario testing")
            else:
                print(f"Warning: {model_path} not found, skipping {model_name}")

        # Results storage
        self.results = []
        self.results_file = '../results/scenario_results.csv'

    def run_scenarios(self):
        print("Starting scenario testing...")
        print("Testing Tesla autopilot failure modes")

        # Create results directory
        os.makedirs('../results', exist_ok=True)

        if not self.loaded_models:
            print("No models found to test. Please train models first.")
            return

        # Define scenarios
        scenarios = [
            self.scenario_phantom_braking,
            self.scenario_cut_in_vehicle,
            self.scenario_adverse_weather,
            self.scenario_sudden_braking
        ]

        # Run each scenario 5 times per model
        for scenario_func in scenarios:
            for model_name, model in self.loaded_models.items():
                for run in range(5):
                    print(f"\nRunning {scenario_func.__name__} | Model: {model_name} | Run: {run+1}/5")
                    results = scenario_func(model, model_name, run)
                    self.results.append(results)

        # Save results
        self.save_results()
        print(f"\nScenario testing complete! Results saved to {self.results_file}")

    def scenario_phantom_braking(self, model, model_name, run):
        """Scenario 1: Phantom braking test - Clear road, no obstacles"""
        town_name = 'Town01'
        weather = carla.WeatherParameters.ClearNoon

        return self.run_scenario(town_name, weather, model, model_name, run, 'phantom_braking')

    def scenario_cut_in_vehicle(self, model, model_name, run):
        """Scenario 2: Cut-in vehicle - Vehicle merges into ego lane"""
        town_name = 'Town01'
        weather = carla.WeatherParameters.ClearNoon

        return self.run_scenario(town_name, weather, model, model_name, run, 'cut_in_vehicle')

    def scenario_adverse_weather(self, model, model_name, run):
        """Scenario 3: Adverse weather - Heavy rain + fog"""
        town_name = 'Town01'
        weather = carla.WeatherParameters.WetNoon  # Rain + fog combined

        return self.run_scenario(town_name, weather, model, model_name, run, 'adverse_weather')

    def scenario_sudden_braking(self, model, model_name, run):
        """Scenario 4: Sudden braking - Vehicle ahead brakes hard"""
        town_name = 'Town01'
        weather = carla.WeatherParameters.ClearNoon

        return self.run_scenario(town_name, weather, model, model_name, run, 'sudden_braking')

    def run_scenario(self, town_name, weather, model, model_name, run, scenario_type):
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
            return self.create_empty_results(town_name, weather, model_name, run, scenario_type)

        # Setup camera sensor
        camera = self.setup_camera(world, vehicle)
        if not camera:
            return self.create_empty_results(town_name, weather, model_name, run, scenario_type)

        # Setup scenario-specific elements
        if scenario_type == 'cut_in_vehicle':
            cut_in_vehicle = self.setup_cut_in_vehicle(world, vehicle, tm)
        elif scenario_type == 'sudden_braking':
            lead_vehicle = self.setup_sudden_braking(world, vehicle, tm)

        # Initialize variables
        start_time = time.time()
        collisions = 0
        phantom_brake_events = 0
        total_distance = 0.0
        speed_sum = 0.0
        frame_count = 0
        scenario_triggered = False
        scenario_duration = 0

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
                        # Preprocess image for model
                        image = self.current_frame.resize((224, 224))
                        image_array = np.array(image).astype(np.float32) / 255.0
                        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 224, 224]

                        # Get model prediction
                        with torch.no_grad():
                            speed_normalized = torch.tensor([[speed / 100.0]], dtype=torch.float32)
                            prediction = model(image_tensor, speed_normalized).squeeze().numpy()

                            # Apply model controls
                            control.throttle = np.clip(prediction[0], 0, 1)
                            control.steer = np.clip(prediction[1], -1, 1)
                            control.brake = np.clip(prediction[2], 0, 1)

                            vehicle.apply_control(control)

                    # Check for collisions
                    if hasattr(self, 'collision_counter'):
                        if self.collision_counter > 0:
                            collisions += self.collision_counter
                            self.collision_counter = 0

                    # Check for phantom braking
                    if control.brake > 0.5 and speed < 5 and not self.is_obstacle_near(world, vehicle):
                        phantom_brake_events += 1

                    # Scenario-specific logic
                    if scenario_type == 'cut_in_vehicle':
                        scenario_triggered = self.handle_cut_in_vehicle(cut_in_vehicle, vehicle)
                    elif scenario_type == 'sudden_braking':
                        scenario_triggered = self.handle_sudden_braking(lead_vehicle, vehicle)

                    # Update metrics
                    frame_count += 1
                    speed_sum += speed
                    total_distance += velocity.x * 0.05  # Approximate distance

                    # Track scenario duration
                    if scenario_triggered:
                        scenario_duration += 0.05  # delta_seconds = 0.05

                else:
                    break  # Vehicle destroyed

        except Exception as e:
            print(f"Scenario exception: {e}")
        finally:
            # Cleanup
            if hasattr(self, 'camera') and self.camera and self.camera.is_alive:
                self.camera.destroy()
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
            if scenario_type == 'cut_in_vehicle' and 'cut_in_vehicle' in locals():
                cut_in_vehicle.destroy()
            if scenario_type == 'sudden_braking' and 'lead_vehicle' in locals():
                lead_vehicle.destroy()

            # Disable synchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)

        # Calculate final metrics
        episode_time = time.time() - start_time
        avg_speed = speed_sum / frame_count if frame_count > 0 else 0
        success = collisions == 0

        # Create results dictionary
        results = {
            'model': model_name,
            'scenario': scenario_type,
            'run': run,
            'town': town_name,
            'weather': 'clear' if weather.weather == carla.WeatherParameters.ClearNoon else 'adverse',
            'collisions': collisions,
            'phantom_brake_events': phantom_brake_events,
            'scenario_duration_sec': scenario_duration,
            'scenario_success': success,
            'average_speed_kmh': avg_speed,
            'total_distance_km': total_distance / 1000.0
        }

        print(f"Scenario results: {results}")
        return results

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

    def setup_cut_in_vehicle(self, world, ego_vehicle, tm):
        """Setup a vehicle that will cut into the ego lane"""
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.audi.tt')

        # Spawn vehicle to the right of ego vehicle
        ego_location = ego_vehicle.get_location()
        spawn_point = carla.Transform(
            carla.Location(x=ego_location.x + 10, y=ego_location.y + 5, z=ego_location.z),
            carla.Rotation(yaw=270)  # Facing left towards ego lane
        )

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, tm.get_port())
            tm.distance_to_leading_vehicle(vehicle, 3.0)
            tm.ignore_lights_percentage(vehicle, 0.0)
            tm.ignore_signs_percentage(vehicle, 0.0)
            tm.ignore_vehicles_percentage(vehicle, 0.0)

            # Plan a path that cuts into ego lane
            waypoints = world.get_map().get_waypoint(vehicle.get_location())
            target_waypoint = world.get_map().get_waypoint(carla.Location(x=ego_location.x, y=ego_location.y - 2))
            vehicle.set_autopilot(True, tm.get_port())

        return vehicle

    def setup_sudden_braking(self, world, ego_vehicle, tm):
        """Setup a vehicle ahead that will brake suddenly"""
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.bmw.grandtourer')

        # Spawn vehicle ahead of ego vehicle
        ego_location = ego_vehicle.get_location()
        spawn_point = carla.Transform(
            carla.Location(x=ego_location.x + 20, y=ego_location.y, z=ego_location.z),
            carla.Rotation(yaw=ego_vehicle.get_transform().rotation.yaw)
        )

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, tm.get_port())
            tm.distance_to_leading_vehicle(vehicle, 3.0)
            tm.ignore_lights_percentage(vehicle, 0.0)
            tm.ignore_signs_percentage(vehicle, 0.0)
            tm.ignore_vehicles_percentage(vehicle, 0.0)

            # Start with normal speed, then brake suddenly
            def brake_suddenly():
                time.sleep(3)  # Wait 3 seconds
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

            import threading
            threading.Thread(target=brake_suddenly, daemon=True).start()

        return vehicle

    def handle_cut_in_vehicle(self, cut_in_vehicle, ego_vehicle):
        """Check if cut-in vehicle has merged into ego lane"""
        if not cut_in_vehicle or not cut_in_vehicle.is_alive:
            return False

        ego_location = ego_vehicle.get_location()
        cut_location = cut_in_vehicle.get_location()
        distance = ego_location.distance(cut_location)

        # If vehicle is within 5m laterally and ahead, consider it merged
        if abs(cut_location.y - ego_location.y) < 5 and cut_location.x < ego_location.x:
            return True

        return False

    def handle_sudden_braking(self, lead_vehicle, ego_vehicle):
        """Check if lead vehicle is braking"""
        if not lead_vehicle or not lead_vehicle.is_alive:
            return False

        lead_control = lead_vehicle.get_control()
        if lead_control.brake > 0.5:
            return True

        return False

    def is_obstacle_near(self, world, vehicle):
        """Check for obstacles within 15m"""
        location = vehicle.get_location()

        # Check for nearby vehicles
        actors = world.get_actors()
        for actor in actors:
            if actor.type_id.startswith('vehicle.'):
                other_location = actor.get_location()
                distance = location.distance(other_location)
                if distance < 15:
                    return True
        return False

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb_image = array[:, :, :3]
        self.current_frame = Image.fromarray(rgb_image)

    def create_empty_results(self, town_name, weather, model_name, run, scenario_type):
        return {
            'model': model_name,
            'scenario': scenario_type,
            'run': run,
            'town': town_name,
            'weather': 'clear' if weather.weather == carla.WeatherParameters.ClearNoon else 'adverse',
            'collisions': 0,
            'phantom_brake_events': 0,
            'scenario_duration_sec': 0,
            'scenario_success': False,
            'average_speed_kmh': 0.0,
            'total_distance_km': 0.0
        }

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        print(f"Saved {len(self.results)} scenario results to {self.results_file}")

if __name__ == "__main__":
    tester = ScenarioTester()
    tester.run_scenarios()