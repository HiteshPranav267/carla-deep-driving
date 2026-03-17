import carla
import torch
import numpy as np
import pandas as pd
import os
import time
import glob
import random
import traceback
import cv2
from collections import deque
from torchvision import transforms
from model import create_model
from PIL import Image


# ---------------------------------------------------------------------------
# Ensemble wrapper – averages predictions from N bagged members
# ---------------------------------------------------------------------------
class EnsembleWrapper:
    def __init__(self, models):
        self.models = models
    def eval(self):
        for m in self.models:
            m.eval()
    def __call__(self, *args, **kwargs):
        preds = [m(*args, **kwargs) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)

# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
class CarlaEvaluator:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60.0)

        # ---- Evaluation matrix ----
        self.evaluation_towns = ['Town03', 'Town05']
        self.weather_conditions = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.WetNoon
        ]
        self.models_to_eval = ['baseline_cnn', 'cnn_lstm']
        self.episodes_per_combo = 10

        # ---- Runtime config / toggles ----
        self.show_live_view = True
        self.screen_width = 800
        self.screen_height = 600
        self.live_view_fps_limit = 20
        self.traffic_vehicles_count = 15
        self.traffic_walkers_count = 20
        self.traffic_seed = 42
        self.no_rendering = False

        # ---- Control mode: 'full_model' or 'assist_throttle' ----
        self.control_mode = 'assist_throttle'

        # ---- Throttle-assist tuning parameters ----
        self.assist_speed_low = 18.0         # km/h – below this, force throttle
        self.assist_speed_high = 28.0        # km/h – above this, cut throttle
        self.assist_throttle_launch = 0.35   # throttle floor when too slow
        self.assist_throttle_cruise = 0.20   # gentle hold throttle in band
        self.launch_protect_speed = 12.0    # km/h – below this, prioritize getting moving
        self.launch_min_throttle = 0.55    # stronger launch floor to break static friction
        self.launch_brake_cap = 0.00         # suppress non-emergency brake during launch
        self.force_launch_duration = 3.0     # sec – hard launch window after episode start
        self.force_launch_speed = 2.0        # km/h – if below this, force stronger throttle
        self.force_launch_throttle = 0.75    # strong kick to ensure vehicle starts moving
        self.brake_deadzone = 0.12           # model brake below this → 0
        self.brake_scale = 1.2               # amplify meaningful brakes
        self.brake_conflict_threshold = 0.55 # brake above this can cancel throttle
        self.emergency_brake_speed = 30.0    # km/h – obstacle override threshold
        self.emergency_brake_value = 0.8     # brake floor during emergency

        # ---- Paths (relative to this script, not CWD) ----
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_file = os.path.join(script_dir, '..', 'results', 'evaluation_log.csv')

        # ---- Preprocessing (must match training) ----
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # ---- Per-episode state ----
        self.collision_counter = 0
        self.lane_invasion_counter = 0
        self.current_frame = None
        self.current_display_frame = None
        self.results = []

        # ---- Per-world-block traffic actor lists ----
        self._traffic_vehicles = []
        self._traffic_walkers = []
        self._walker_controllers = []

        # ---- Live-view bookkeeping ----
        self._live_view_ok = False
        self._last_display_time = 0.0

    # ===================================================================
    #  Top-level entry point
    # ===================================================================
    def evaluate(self):
        random.seed(self.traffic_seed)
        np.random.seed(self.traffic_seed)

        print("Starting evaluation...")
        print(f"  Control mode: {self.control_mode}")
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        loaded_models = {}
        for model_name in self.models_to_eval:
            model = self.load_model_for_evaluation(model_name)
            if model is not None:
                model.eval()
                loaded_models[model_name] = model
            else:
                print(f"Warning: Model {model_name} missing, skipping.")

        if not loaded_models:
            print("No models found. Exiting.")
            return

        for town_idx, town_name in enumerate(self.evaluation_towns):
            for weather_idx, weather in enumerate(self.weather_conditions):
                world = self.setup_world_block(town_name, weather)
                if world is None:
                    continue

                try:
                    for model_name, model in loaded_models.items():
                        print(f"\n--- Testing {model_name} in {town_name} "
                              f"| Weather {'clear' if weather_idx == 0 else 'adverse'} "
                              f"| Mode: {self.control_mode} ---")
                        for episode in range(self.episodes_per_combo):
                            try:
                                result = self.run_episode(
                                    world, model, model_name,
                                    episode, town_idx, weather_idx
                                )
                                self.results.append(result)
                                self.save_results()
                            except Exception as e:
                                print(f"Exception in episode {episode}: {e}")
                                traceback.print_exc()
                                weather_str = 'clear' if weather_idx == 0 else 'adverse'
                                self.results.append(
                                    self.create_empty_results(
                                        town_name, weather_str, model_name, episode
                                    )
                                )
                                self.save_results()
                finally:
                    self.teardown_world_block(world)

        self._destroy_live_view()
        print(f"\nEvaluation complete! {len(self.results)} rows saved to {self.results_file}")

    # ===================================================================
    #  World-block lifecycle
    # ===================================================================
    def setup_world_block(self, town_name, weather):
        print(f"Loading {town_name}...")
        for attempt in range(3):
            try:
                self.client.load_world(town_name)
                time.sleep(2)
                world = self.client.get_world()
                world.set_weather(weather)

                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                if self.no_rendering:
                    settings.no_rendering_mode = True
                world.apply_settings(settings)

                tm = self.client.get_trafficmanager()
                tm.set_synchronous_mode(True)
                tm.set_random_device_seed(self.traffic_seed)

                self.spawn_traffic(world, tm)

                for _ in range(10):
                    world.tick()

                return world
            except Exception as e:
                print(f"Failed to load world (attempt {attempt + 1}/3): {e}")
                time.sleep(5)
        return None

    def teardown_world_block(self, world):
        self.destroy_traffic()
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(False)
        except Exception as e:
            print(f"Error tearing down world block: {e}")

    # ===================================================================
    #  Traffic generation & cleanup
    # ===================================================================
    def spawn_traffic(self, world, tm):
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle_bps = bp_lib.filter('vehicle.*')
        vehicle_bps = [bp for bp in vehicle_bps
                       if int(bp.get_attribute('number_of_wheels')) == 4]

        n_vehicles = min(self.traffic_vehicles_count, len(spawn_points) - 1)
        for i in range(n_vehicles):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            try:
                npc = world.try_spawn_actor(bp, spawn_points[i])
                if npc is not None:
                    npc.set_autopilot(True, tm.get_port())
                    self._traffic_vehicles.append(npc)
            except Exception:
                pass

        print(f"  Spawned {len(self._traffic_vehicles)} NPC vehicles")

        walker_bps = bp_lib.filter('walker.pedestrian.*')
        spawned_walkers = 0

        for _ in range(self.traffic_walkers_count):
            bp = random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            spawn_t = carla.Transform(location=loc)
            try:
                walker = world.try_spawn_actor(bp, spawn_t)
                if walker is None:
                    continue
                self._traffic_walkers.append(walker)
                spawned_walkers += 1
            except Exception:
                continue

        world.tick()

        walker_ctrl_bp = bp_lib.find('controller.ai.walker')
        for walker in self._traffic_walkers:
            try:
                ctrl = world.spawn_actor(walker_ctrl_bp, carla.Transform(), attach_to=walker)
                self._walker_controllers.append(ctrl)
            except Exception:
                self._walker_controllers.append(None)

        world.tick()
        for ctrl in self._walker_controllers:
            if ctrl is not None:
                try:
                    ctrl.start()
                    dest = world.get_random_location_from_navigation()
                    if dest:
                        ctrl.go_to_location(dest)
                    ctrl.set_max_speed(1.4)
                except Exception:
                    pass

        print(f"  Spawned {spawned_walkers} NPC walkers")

    def destroy_traffic(self):
        for ctrl in self._walker_controllers:
            if ctrl is not None:
                try:
                    ctrl.stop()
                    ctrl.destroy()
                except Exception:
                    pass
        self._walker_controllers.clear()

        for w in self._traffic_walkers:
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass
        self._traffic_walkers.clear()

        for v in self._traffic_vehicles:
            if v is not None:
                try:
                    v.destroy()
                except Exception:
                    pass
        self._traffic_vehicles.clear()

    # ===================================================================
    #  Model loading (ensemble-aware)
    # ===================================================================
    def load_model_for_evaluation(self, model_name):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        legacy_path = os.path.join(model_dir, f"{model_name}.pth")

        ensemble_pattern = os.path.join(model_dir, f"{model_name}_member_*.pth")
        ensemble_files = glob.glob(ensemble_pattern)

        if ensemble_files:
            ensemble_files.sort()
            print(f"Loading {len(ensemble_files)} ensemble members for {model_name}...")
            models = []
            for ckpt_path in ensemble_files:
                try:
                    m = create_model(model_name)
                    m.load_state_dict(
                        torch.load(ckpt_path, map_location='cpu', weights_only=True)
                    )
                    models.append(m)
                except Exception as e:
                    print(f"Failed to load member {ckpt_path}: {e}")
            if models:
                return EnsembleWrapper(models)
            return None

        try:
            model = create_model(model_name)
        except Exception as e:
            print(f"Failed to create model {model_name}: {e}")
            return None

        if os.path.exists(legacy_path):
            print(f"Loaded legacy checkpoint from {legacy_path}")
            model.load_state_dict(
                torch.load(legacy_path, map_location='cpu', weights_only=True)
            )
            return model

        print(f"Warning: No checkpoints found for {model_name}")
        return None

    # ===================================================================
    #  Ego vehicle & sensors
    # ===================================================================
    def spawn_ego_vehicle(self, world):
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for sp in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                vehicle.set_autopilot(False)
                return vehicle, sp
        return None, None

    def setup_sensors(self, world, vehicle):
        self.current_frame = None
        self.current_display_frame = None
        self.collision_counter = 0
        self.lane_invasion_counter = 0

        bp_lib = world.get_blueprint_library()

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.screen_width))
        cam_bp.set_attribute('image_size_y', str(self.screen_height))
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

        def process_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            bgr = array[:, :, :3].copy()
            rgb = bgr[:, :, ::-1].copy()
            self.current_frame = Image.fromarray(rgb)
            self.current_display_frame = bgr

        camera.listen(process_image)

        col_bp = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)

        def process_collision(event):
            self.collision_counter += 1

        collision_sensor.listen(process_collision)

        lane_bp = bp_lib.find('sensor.other.lane_invasion')
        lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=vehicle)

        def process_lane_invasion(event):
            self.lane_invasion_counter += 1

        lane_sensor.listen(process_lane_invasion)

        return [camera, collision_sensor, lane_sensor]

    # ===================================================================
    #  Control arbitration layer
    # ===================================================================
    def arbitrate_control(self, raw_throttle, raw_steer, raw_brake,
                          speed, obstacle_near):
        """
        Takes raw model outputs and returns final control values plus
        diagnostic info.  In 'full_model' mode this is mostly pass-through.
        In 'assist_throttle' mode the pipeline is:
          1. Deadzone on brake
          2. Scale brake
          3. Throttle assist (launch / cruise / overspeed)
          4. Brake-throttle conflict resolution
          5. Emergency obstacle override
          6. Final clip + float cast
        Returns: (throttle, steer, brake, assist_reason, is_emergency)
        """
        steer = float(np.clip(raw_steer, -1.0, 1.0))

        if self.control_mode == 'full_model':
            throttle = float(np.clip(raw_throttle, 0.0, 1.0))
            brake = float(np.clip(raw_brake, 0.0, 1.0))
            return throttle, steer, brake, 'full_model', False

        # --- assist_throttle mode ---

        # 1. Brake deadzone: ignore tiny model brake noise
        brake = raw_brake if raw_brake >= self.brake_deadzone else 0.0

        # 2. Scale meaningful brakes
        brake = brake * self.brake_scale

        # 3. Throttle assist with low-speed launch protection
        assist_reason = ''
        launch_protect_active = (speed < self.launch_protect_speed) and (not obstacle_near)
        if launch_protect_active:
            throttle = max(raw_throttle, self.assist_throttle_launch, self.launch_min_throttle)
            brake = min(brake, self.launch_brake_cap)
            assist_reason = 'launch_protect'
        elif speed < self.assist_speed_low:
            throttle = max(raw_throttle, self.assist_throttle_launch)
            assist_reason = 'launch'
        elif speed <= self.assist_speed_high:
            throttle = max(raw_throttle, self.assist_throttle_cruise)
            assist_reason = 'cruise'
        else:
            throttle = 0.0
            assist_reason = 'overspeed'

        # 4. Brake-throttle conflict resolver
        if brake > self.brake_conflict_threshold and not launch_protect_active:
            # Meaningful brake → cut throttle entirely
            throttle = 0.0
            if not assist_reason:
                assist_reason = 'braking'
        elif throttle > 0.5:
            # High throttle → suppress light brake unless emergency
            brake = min(brake, 0.05)

        # 5. Emergency obstacle override
        is_emergency = False
        if obstacle_near and speed > self.emergency_brake_speed:
            brake = max(brake, self.emergency_brake_value)
            throttle = 0.0
            assist_reason = 'emergency'
            is_emergency = True

        # 6. Final clip + cast
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        return throttle, steer, brake, assist_reason, is_emergency

    # ===================================================================
    #  Live-view display
    # ===================================================================
    WINDOW_NAME = 'CARLA Evaluation - Live View'

    def _init_live_view(self):
        if not self.show_live_view or self.no_rendering:
            self._live_view_ok = False
            return
        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.screen_width, self.screen_height)
            self._live_view_ok = True
        except Exception as e:
            print(f"Warning: Could not open live view window ({e}). "
                  "Continuing without display.")
            self._live_view_ok = False

    def _update_live_view(self, hud_info):
        if not self._live_view_ok:
            return

        now = time.time()
        min_interval = 1.0 / max(1, self.live_view_fps_limit)
        if (now - self._last_display_time) < min_interval:
            return
        self._last_display_time = now

        frame = self.current_display_frame
        if frame is None:
            return

        display = frame.copy()

        # Semi-transparent dark banner (taller to fit assist info)
        overlay = display.copy()
        banner_h = 280 if self.control_mode == 'assist_throttle' else 180
        cv2.rectangle(overlay, (0, 0), (self.screen_width, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        green = (0, 255, 0)
        yellow = (0, 255, 255)
        red = (0, 0, 255)
        white = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1

        y = 25
        def put(text, color=green):
            nonlocal y
            cv2.putText(display, text, (15, y), font, scale, color, thick, cv2.LINE_AA)
            y += 26

        put(f"Mode: {self.control_mode.upper()}", yellow)
        put(f"Model: {hud_info.get('model', '?')}")
        put(f"Town: {hud_info.get('town', '?')}  |  "
            f"Weather: {hud_info.get('weather', '?')}")
        put(f"Episode: {hud_info.get('episode', '?')}  |  "
            f"Speed: {hud_info.get('speed', 0):.1f} km/h")
        put(f"Collisions: {hud_info.get('collisions', 0)}  |  "
            f"Lane Inv: {hud_info.get('lane_invasions', 0)}")
        put(f"Route: {hud_info.get('route_completion', 0):.1%}  |  "
            f"Time: {hud_info.get('elapsed', 0):.1f}s")

        if self.control_mode == 'assist_throttle':
            rt = hud_info.get('raw_throttle', 0)
            rs = hud_info.get('raw_steer', 0)
            rb = hud_info.get('raw_brake', 0)
            ft = hud_info.get('final_throttle', 0)
            fs = hud_info.get('final_steer', 0)
            fb = hud_info.get('final_brake', 0)
            reason = hud_info.get('assist_reason', '')

            put(f"Raw   T:{rt:.2f}  S:{rs:.2f}  B:{rb:.2f}", white)
            put(f"Final T:{ft:.2f}  S:{fs:.2f}  B:{fb:.2f}", yellow)

            reason_color = red if reason == 'emergency' else yellow
            put(f"Assist: {reason}", reason_color)

        cv2.imshow(self.WINDOW_NAME, display)
        cv2.waitKey(1)

    def _destroy_live_view(self):
        if self._live_view_ok:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
                cv2.waitKey(1)
            except Exception:
                pass
            self._live_view_ok = False

    # ===================================================================
    #  Episode runner
    # ===================================================================
    def run_episode(self, world, model, model_name, episode, town_idx, weather_idx):
        weather_str = 'clear' if weather_idx == 0 else 'adverse'
        town_name = self.evaluation_towns[town_idx]
        print(f"  Episode {episode + 1}/{self.episodes_per_combo}")

        vehicle = None
        sensors = []

        try:
            vehicle, start_transform = self.spawn_ego_vehicle(world)
            if vehicle is None:
                print("  Failed to spawn vehicle.")
                return self.create_empty_results(town_name, weather_str, model_name, episode)

            sensors = self.setup_sensors(world, vehicle)
            self._init_live_view()

            for _ in range(5):
                world.tick()

            start_time = time.time()
            max_seconds = 60.0

            collisions = 0
            lane_invasions = 0
            phantom_brake_events = 0
            total_distance = 0.0
            speed_sum = 0.0
            frame_count = 0
            frame_buffer = deque(maxlen=5)

            # Assist-mode accumulators
            inference_ticks = 0
            sum_applied_throttle = 0.0
            sum_model_brake = 0.0
            sum_applied_brake = 0.0
            throttle_assist_active_ticks = 0
            emergency_override_count = 0

            # Per-tick HUD state (kept outside loop for live view)
            last_hud = {}

            while (time.time() - start_time) < max_seconds:
                world.tick()

                if not vehicle.is_alive:
                    break

                velocity = vehicle.get_velocity()
                speed = float(np.sqrt(
                    velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6)

                # Defaults for ticks without inference
                raw_t = raw_s = raw_b = 0.0
                final_t = final_s = final_b = 0.0
                assist_reason = ''

                if self.current_frame is not None:
                    frame_tensor = self.eval_transform(self.current_frame)
                    frame_buffer.append(frame_tensor)

                    if model_name == 'cnn_lstm' and len(frame_buffer) < 5:
                        frame_count += 1
                        speed_sum += speed
                        total_distance += float(np.sqrt(
                            velocity.x**2 + velocity.y**2 + velocity.z**2) * 0.05)
                        continue

                    with torch.no_grad():
                        speed_normalized = torch.tensor(
                            [[speed / 100.0]], dtype=torch.float32
                        )
                        if model_name == 'cnn_lstm':
                            image_tensor = torch.stack(
                                list(frame_buffer), dim=0
                            ).unsqueeze(0)
                        else:
                            image_tensor = frame_buffer[-1].unsqueeze(0)

                        prediction = model(
                            image_tensor, speed_normalized
                        ).squeeze().numpy()

                        raw_t = float(prediction[0])
                        raw_s = float(prediction[1])
                        raw_b = float(prediction[2])

                        obstacle_near = self.is_obstacle_near(world, vehicle)

                        final_t, final_s, final_b, assist_reason, is_emergency = \
                            self.arbitrate_control(
                                raw_t, raw_s, raw_b, speed, obstacle_near
                            )

                        # Hard launch override: guarantee initial movement unless blocked.
                        elapsed = time.time() - start_time
                        if (self.control_mode == 'assist_throttle'
                                and elapsed < self.force_launch_duration
                                and speed < self.force_launch_speed
                                and not obstacle_near):
                            final_t = max(final_t, self.force_launch_throttle)
                            final_b = 0.0
                            assist_reason = 'force_launch'

                        ctrl = carla.VehicleControl()
                        ctrl.throttle = final_t
                        ctrl.steer = final_s
                        ctrl.brake = final_b
                        ctrl.hand_brake = False
                        ctrl.reverse = False
                        ctrl.manual_gear_shift = False
                        vehicle.apply_control(ctrl)
                        inference_ticks += 1

                        # Phantom brake detection
                        if (final_b > 0.5 and speed < 5
                                and not obstacle_near):
                            phantom_brake_events += 1

                        # Accumulate assist metrics
                        sum_applied_throttle += final_t
                        sum_model_brake += float(np.clip(raw_b, 0, 1))
                        sum_applied_brake += final_b
                        if is_emergency:
                            emergency_override_count += 1
                        if self.control_mode == 'assist_throttle' and assist_reason:
                            throttle_assist_active_ticks += 1

                # Drain counters
                if self.collision_counter > 0:
                    collisions += self.collision_counter
                    self.collision_counter = 0

                if self.lane_invasion_counter > 0:
                    lane_invasions += self.lane_invasion_counter
                    self.lane_invasion_counter = 0

                frame_count += 1
                speed_sum += speed
                total_distance += float(np.sqrt(
                    velocity.x**2 + velocity.y**2 + velocity.z**2) * 0.05)

                # Route completion (running)
                if vehicle.is_alive:
                    cur_t = vehicle.get_transform()
                    dist_from_start = float(np.sqrt(
                        (cur_t.location.x - start_transform.location.x)**2 +
                        (cur_t.location.y - start_transform.location.y)**2
                    ))
                    route_completion = min(dist_from_start / 1000.0, 1.0)
                else:
                    route_completion = 0.0

                # Live view
                last_hud = {
                    'model': model_name,
                    'town': town_name,
                    'weather': weather_str,
                    'episode': f"{episode + 1}/{self.episodes_per_combo}",
                    'speed': speed,
                    'collisions': collisions,
                    'lane_invasions': lane_invasions,
                    'route_completion': route_completion,
                    'elapsed': time.time() - start_time,
                    'raw_throttle': raw_t,
                    'raw_steer': raw_s,
                    'raw_brake': raw_b,
                    'final_throttle': final_t,
                    'final_steer': final_s,
                    'final_brake': final_b,
                    'assist_reason': assist_reason,
                }
                self._update_live_view(last_hud)

            # ---- Final metrics ----
            if vehicle and vehicle.is_alive:
                cur_t = vehicle.get_transform()
                distance_traveled = float(np.sqrt(
                    (cur_t.location.x - start_transform.location.x)**2 +
                    (cur_t.location.y - start_transform.location.y)**2
                ))
                route_completion = float(min(distance_traveled / 1000.0, 1.0))
            else:
                route_completion = 0.0

            avg_speed = float(speed_sum / max(1, frame_count))
            success = bool(route_completion > 0.9 and collisions == 0)

            infer_ticks = max(1, inference_ticks)  # only ticks with actual model output

            return {
                'model': str(model_name),
                'town': str(town_name),
                'weather': str(weather_str),
                'episode': int(episode),
                'control_mode': str(self.control_mode),
                'route_completion': float(route_completion),
                'collisions': int(collisions),
                'lane_invasions': int(lane_invasions),
                'phantom_brake_events': int(phantom_brake_events),
                'average_speed_kmh': float(avg_speed),
                'episode_time_sec': float(time.time() - start_time),
                'total_distance_km': float(total_distance / 1000.0),
                'avg_applied_throttle': float(sum_applied_throttle / infer_ticks),
                'avg_model_brake': float(sum_model_brake / infer_ticks),
                'avg_applied_brake': float(sum_applied_brake / infer_ticks),
                'throttle_assist_active_ratio': float(
                    throttle_assist_active_ticks / infer_ticks),
                'emergency_override_count': int(emergency_override_count),
                'success': success,
            }

        finally:
            self.cleanup_episode(vehicle, sensors)

    # ===================================================================
    #  Cleanup helpers
    # ===================================================================
    def cleanup_episode(self, vehicle, sensors):
        self._destroy_live_view()

        for sensor in sensors:
            if sensor is not None and sensor.is_alive:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
        if vehicle is not None and vehicle.is_alive:
            try:
                vehicle.destroy()
            except Exception:
                pass

        self.current_frame = None
        self.current_display_frame = None
        self.collision_counter = 0
        self.lane_invasion_counter = 0

    def is_obstacle_near(self, world, vehicle):
        if vehicle is None or not vehicle.is_alive:
            return False
        location = vehicle.get_location()
        for actor in world.get_actors():
            if actor.id != vehicle.id and actor.type_id.startswith('vehicle.'):
                if location.distance(actor.get_location()) < 15:
                    return True
        return False

    # ===================================================================
    #  Results helpers
    # ===================================================================
    def create_empty_results(self, town_name, weather_str, model_name, episode):
        return {
            'model': str(model_name),
            'town': str(town_name),
            'weather': str(weather_str),
            'episode': int(episode),
            'control_mode': str(self.control_mode),
            'route_completion': 0.0,
            'collisions': 0,
            'lane_invasions': 0,
            'phantom_brake_events': 0,
            'average_speed_kmh': 0.0,
            'episode_time_sec': 0.0,
            'total_distance_km': 0.0,
            'avg_applied_throttle': 0.0,
            'avg_model_brake': 0.0,
            'avg_applied_brake': 0.0,
            'throttle_assist_active_ratio': 0.0,
            'emergency_override_count': 0,
            'success': False,
        }

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)


# ===================================================================
#  Entry point
# ===================================================================
if __name__ == "__main__":
    evaluator = CarlaEvaluator()
    evaluator.evaluate()