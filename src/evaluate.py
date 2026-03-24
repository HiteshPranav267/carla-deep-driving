import carla
import torch
import numpy as np
import pandas as pd
import os
import time
import math
import glob
import random
import traceback
import cv2
from collections import deque
from torchvision import transforms
from model import create_model
from PIL import Image

# Route planner for junction navigation
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    HAS_ROUTE_PLANNER = True
except ImportError:
    HAS_ROUTE_PLANNER = False
    print("⚠️  GlobalRoutePlanner not available. Falling back to anchor-based tracking.")


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
        self.models_to_eval = ['baseline_cnn', 'cnn_gru', 'gru_only']
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

        # ---- Lane-first controller (Stanley) ----
        self.lane_assist_enabled = True
        self.stanley_k_heading = 1.0         # heading error gain
        self.stanley_k_crosstrack = 1.4      # cross-track gain (reduced from 2.5)
        self.stanley_softening = 1.0         # v + epsilon denominator (km/h)
        self.stanley_sign = 1.0              # steering sign convention (+1 or -1)
        self.max_lane_steer = 0.50           # allow enough steering authority for turns
        self.model_steer_max = 0.10          # max absolute model steer correction
        self.lane_weight = 1.0               # lane tracker weight (primary)
        self.model_steer_weight = 0.0        # model steer disabled in assist mode
        self.steer_smoothing_alpha = 0.7     # less damping so steering reacts sooner
        self.steer_rate_limit = 0.08         # allow faster steer changes per tick
        self.lateral_deadband = 0.05         # m – ignore lateral error below this
        self.warmup_lane_only_sec = 4.0      # pure lane-only during first N seconds
        self.force_launch_steer_clamp = 0.06 # max |steer| during force launch (was 0.10)
        self._prev_final_steer = 0.0
        self._prev_lane_steer = 0.0

        # ---- Off-lane safety mode ----
        self.offlane_lat_threshold = 1.5     # m – lateral error to trigger safety
        self.offlane_hdg_threshold = math.radians(35)  # rad
        self.offlane_throttle_cap = 0.20
        self.offlane_brake_floor = 0.05
        self.offlane_steer_cap = 0.30
        self.offlane_exit_ticks = 10         # must be below thresholds this many ticks

        # ---- Speed-steer coupling ----
        self.steer_throttle_low = 0.22       # |steer| > this → cap throttle to 0.18
        self.steer_throttle_high = 0.30      # |steer| > this → cap throttle to 0.10

        # ---- Lane-invasion recovery ----
        self.lane_invasion_recovery_sec = 3.0
        self.lane_recovery_speed_target = 15.0
        self.lane_recovery_lookahead_bonus = 3.0

        # ---- Route-based navigation state ----
        self._route_planner = None
        self._route_wp_seq = []  # waypoint sequence for this episode
        self._route_wp_idx = 0   # current index in route
        self._lane_confidence = 1.0
        self._last_stable_steer = 0.0
        
        # ---- Fallback anchor-based tracking (if route planner unavailable) ----
        self._anchor_wp = None
        self._anchor_road_id = -1
        self._anchor_lane_id = 0

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

                # Initialize route planner
                if HAS_ROUTE_PLANNER:
                    carla_map = world.get_map()
                    self._route_planner = GlobalRoutePlanner(carla_map, 2.0)
                    print(f"  ✓ Route planner initialized for {town_name}")
                else:
                    self._route_planner = None
                    print(f"  ⚠️  Route planner unavailable; using fallback anchor-based tracking")

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
    #  Lane-anchor tracking + error estimation
    # ===================================================================
    def _best_waypoint_continuity(self, candidates, ego_fwd,
                                   preferred_road_id, preferred_lane_id):
        """
        Pick candidate preferring same road_id/lane_id, fallback to
        best heading alignment.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # First: try candidates matching road_id + lane_id
        same_lane = [wp for wp in candidates
                     if wp.road_id == preferred_road_id
                     and wp.lane_id == preferred_lane_id]
        if same_lane:
            candidates = same_lane

        # Among remaining, pick best heading alignment
        best_dot = -999.0
        best = candidates[0]
        for wp in candidates:
            fwd = wp.transform.get_forward_vector()
            d = ego_fwd.x * fwd.x + ego_fwd.y * fwd.y
            if d > best_dot:
                best_dot = d
                best = wp
        return best

    def _advance_anchor(self, ego_loc, ego_fwd):
        """
        Advance the lane anchor waypoint along the road.
        Uses next(3m) from current anchor, preferring same road_id/lane_id.
        Snaps to ego when distance is too large (respawn scenario).
        """
        if self._anchor_wp is None:
            return

        anchor_loc = self._anchor_wp.transform.location
        dist_sq = ((ego_loc.x - anchor_loc.x)**2 +
                   (ego_loc.y - anchor_loc.y)**2)

        # If ego is too far from anchor (>20m), re-snap (probably respawned)
        if dist_sq > 400.0:
            new_wp = self._anchor_wp  # keep current, get_lane_errors will re-snap
            return

        # Advance anchor if ego has passed it (dot product with anchor fwd)
        anchor_fwd = self._anchor_wp.transform.get_forward_vector()
        dx = ego_loc.x - anchor_loc.x
        dy = ego_loc.y - anchor_loc.y
        along = dx * anchor_fwd.x + dy * anchor_fwd.y

        if along > 1.5:  # ego is >1.5m ahead of anchor
            cands = self._anchor_wp.next(min(along + 1.0, 6.0))
            new_wp = self._best_waypoint_continuity(
                cands, ego_fwd, self._anchor_road_id, self._anchor_lane_id)
            if new_wp is not None:
                # Check lane continuity confidence
                if (new_wp.road_id != self._anchor_road_id or
                        new_wp.lane_id != self._anchor_lane_id):
                    if not new_wp.is_junction:
                        self._lane_confidence = max(0.2, self._lane_confidence - 0.3)
                self._anchor_wp = new_wp
                self._anchor_road_id = new_wp.road_id
                self._anchor_lane_id = new_wp.lane_id

    # ===================================================================
    #  Route-aware navigation
    # ===================================================================
    def _init_episode_route(self, world, vehicle, spawn_transform):
        """Initialize route for this episode using GlobalRoutePlanner."""
        if not HAS_ROUTE_PLANNER or self._route_planner is None:
            self._route_wp_seq = []
            self._route_wp_idx = 0
            return
        
        try:
            spawn_loc = spawn_transform.location
            carla_map = world.get_map()
            
            # Get a far destination (e.g., 500m away waypoint)
            spawn_wp = carla_map.get_waypoint(spawn_loc, project_to_road=True)
            if spawn_wp is None:
                self._route_wp_seq = []
                return
            
            # Pick a far destination by going many waypoints ahead
            dest_wp = spawn_wp
            for _ in range(50):  # ~250m ahead (5m per waypoint)
                next_wps = dest_wp.next(5.0)
                if next_wps:
                    dest_wp = next_wps[0]
                else:
                    break
            
            dest_loc = dest_wp.transform.location
            
            # Compute route using GlobalRoutePlanner A*
            route_list = self._route_planner.trace_route(spawn_loc, dest_loc)
            
            # Extract waypoints with confidence
            self._route_wp_seq = [(wp, cmd) for wp, cmd in route_list]
            self._route_wp_idx = 0
            
            print(f"    Route initialized: {len(self._route_wp_seq)} waypoints")
        except Exception as e:
            print(f"    Route init failed: {e}. Using fallback anchor tracking.")
            self._route_wp_seq = []
            self._route_wp_idx = 0

    def _get_route_next_waypoint(self):
        """Get the next target waypoint from the pre-computed route."""
        if not self._route_wp_seq or self._route_wp_idx >= len(self._route_wp_seq):
            return None
        
        wp, _ = self._route_wp_seq[self._route_wp_idx]
        return wp

    def _advance_route_waypoint(self, ego_loc):
        """Advance route waypoint index if ego is close enough."""
        if not self._route_wp_seq:
            return
        
        next_wp = self._get_route_next_waypoint()
        if next_wp is None:
            return
        
        dist_to_wp = ego_loc.distance(next_wp.transform.location)
        if dist_to_wp < 3.0:  # Move to next waypoint when within 3m
            self._route_wp_idx += 1

    # ===================================================================
    #  Lane error estimation (route-aware)
    # ===================================================================
    def get_lane_errors(self, world, vehicle, extra_lookahead=0.0):
        """
        Route-aware or anchor-based lane error estimation.
        Uses route planner if available, otherwise falls back to anchor tracking.
        Returns: (heading_err_rad, lateral_err_m, is_junction, confidence).
        """
        try:
            ego_t = vehicle.get_transform()
            ego_loc = ego_t.location
            ego_fwd = ego_t.get_forward_vector()
            
            # If route available, use it; otherwise fall back to anchor
            if self._route_wp_seq:
                self._advance_route_waypoint(ego_loc)
                anchor_wp = self._get_route_next_waypoint()
                
                # If route waypoint is available, use it directly
                if anchor_wp is not None:
                    is_junction = anchor_wp.is_junction
                    confidence = 0.95  # High confidence since route is explicit
                    
                    # Target waypoint for heading: lookahead from current anchor
                    v = vehicle.get_velocity()
                    speed_kmh = float(np.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6)
                    if speed_kmh < 15.0:
                        la_dist = 3.5 + extra_lookahead
                    elif speed_kmh < 30.0:
                        la_dist = 5.0 + extra_lookahead
                    else:
                        la_dist = 7.0 + extra_lookahead
                    
                    cands = anchor_wp.next(la_dist)
                    if cands:
                        target_wp = cands[0]
                    else:
                        target_wp = anchor_wp
                    
                    # --- Heading error ---
                    wp_fwd = target_wp.transform.get_forward_vector()
                    cross = ego_fwd.x * wp_fwd.y - ego_fwd.y * wp_fwd.x
                    dot = ego_fwd.x * wp_fwd.x + ego_fwd.y * wp_fwd.y
                    heading_err = math.atan2(cross, dot)
                    
                    # --- Lateral error (cross-track to route waypoint center) ---
                    wp_loc = anchor_wp.transform.location
                    dx = ego_loc.x - wp_loc.x
                    dy = ego_loc.y - wp_loc.y
                    anc_fwd = anchor_wp.transform.get_forward_vector()
                    wp_right_x = anc_fwd.y
                    wp_right_y = -anc_fwd.x
                    lateral_err = dx * wp_right_x + dy * wp_right_y
                    
                    # Anti-oscillation deadband
                    if abs(lateral_err) < self.lateral_deadband:
                        lateral_err = 0.0
                    
                    return float(heading_err), float(lateral_err), bool(is_junction), float(confidence)
            
            # Fallback: use anchor-based tracking
            # Re-snap anchor if missing
            if self._anchor_wp is None:
                carla_map = world.get_map()
                wp = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
                if wp is None:
                    return 0.0, 0.0, False, 0.0
                self._anchor_wp = wp
                self._anchor_road_id = wp.road_id
                self._anchor_lane_id = wp.lane_id

            # Advance anchor along road
            self._advance_anchor(ego_loc, ego_fwd)

            is_junction = self._anchor_wp.is_junction
            confidence = self._lane_confidence
            if is_junction:
                confidence = min(confidence, 0.8)

            # Restore confidence gradually when on stable lane
            if not is_junction and self._lane_confidence < 1.0:
                self._lane_confidence = min(1.0, self._lane_confidence + 0.02)

            # Target waypoint for heading: speed-adaptive lookahead from anchor
            v = vehicle.get_velocity()
            speed_kmh = float(np.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6)
            if speed_kmh < 15.0:
                la_dist = 3.5 + extra_lookahead
            elif speed_kmh < 30.0:
                la_dist = 5.0 + extra_lookahead
            else:
                la_dist = 7.0 + extra_lookahead
            cands = self._anchor_wp.next(la_dist)
            target_wp = self._best_waypoint_continuity(
                cands, ego_fwd, self._anchor_road_id, self._anchor_lane_id)
            if target_wp is None:
                target_wp = self._anchor_wp

            # --- Heading error ---
            wp_fwd = target_wp.transform.get_forward_vector()
            cross = ego_fwd.x * wp_fwd.y - ego_fwd.y * wp_fwd.x
            dot = ego_fwd.x * wp_fwd.x + ego_fwd.y * wp_fwd.y
            heading_err = math.atan2(cross, dot)

            # --- Lateral error (cross-track to anchor center) ---
            wp_loc = self._anchor_wp.transform.location
            dx = ego_loc.x - wp_loc.x
            dy = ego_loc.y - wp_loc.y
            anc_fwd = self._anchor_wp.transform.get_forward_vector()
            wp_right_x = anc_fwd.y
            wp_right_y = -anc_fwd.x
            lateral_err = dx * wp_right_x + dy * wp_right_y

            # Anti-oscillation deadband
            if abs(lateral_err) < self.lateral_deadband:
                lateral_err = 0.0

            return float(heading_err), float(lateral_err), bool(is_junction), float(confidence)

        except Exception:
            return 0.0, 0.0, False, 0.0

    def compute_stanley_steer(self, heading_err, lateral_err, speed_kmh,
                              is_junction, confidence):
        """
        Stanley-style lane controller:
          δ = k_h · e_ψ + arctan(k_c · e_y / (v + ε))
        Returns smoothed + rate-limited lane steer.
        """
        k_h = self.stanley_k_heading
        k_c = self.stanley_k_crosstrack

        # At junctions with route planner, maintain full gains (we know the correct path).
        # At junctions without route planner (anchor fallback), reduce gains slightly.
        if is_junction and not self._route_wp_seq:
            # Anchor-based fallback: gentle junctions
            k_h *= 0.75
            k_c *= 0.65
        # If confidence drops significantly, reduce gains (anchor fallback)
        elif confidence < 0.5:
            scale = max(0.4, confidence)
            k_h *= scale
            k_c *= scale
        # Stanley expects velocity in m/s, not km/h.
        speed_mps = speed_kmh / 3.6
        v = max(speed_mps, self.stanley_softening)  # avoid div-by-zero

        # Stanley law. Sign is configurable because frame conventions
        # can invert steer direction depending on map/controller setup.
        heading_term = k_h * heading_err
        crosstrack_term = math.atan2(k_c * lateral_err, v)
        raw_steer = self.stanley_sign * (heading_term + crosstrack_term)

        # Clip to max
        clipped = float(np.clip(raw_steer, -self.max_lane_steer, self.max_lane_steer))

        # EMA smoothing
        alpha = self.steer_smoothing_alpha
        smoothed = alpha * self._prev_lane_steer + (1.0 - alpha) * clipped
        self._prev_lane_steer = smoothed

        lane_mode = 'lane_tracking'
        return float(smoothed), lane_mode

    def _rate_limit_steer(self, target_steer):
        """Apply max steer change per tick and update state."""
        delta = target_steer - self._prev_final_steer
        if abs(delta) > self.steer_rate_limit:
            delta = math.copysign(self.steer_rate_limit, delta)
        result = self._prev_final_steer + delta
        result = float(np.clip(result, -1.0, 1.0))
        self._prev_final_steer = result
        return result

    # ===================================================================
    #  Control arbitration layer
    # ===================================================================
    def arbitrate_control(self, raw_throttle, raw_steer, raw_brake,
                          speed, obstacle_near,
                          lane_steer=0.0, lane_mode='',
                          elapsed_sec=0.0, is_recovery=False,
                          heading_err=0.0, lateral_err=0.0, status='Following Route'):
        """
        Lane-first arbitration with off-lane safety + speed-steer coupling + red light braking.
        Returns: (throttle, steer, brake, assist_reason, is_emergency,
                  lane_mode, is_offlane)
        """
        model_steer = float(np.clip(raw_steer, -1.0, 1.0))

        if self.control_mode == 'full_model' or not self.lane_assist_enabled:
            steer = model_steer
            lane_mode = 'model_only'
        elif elapsed_sec < self.warmup_lane_only_sec:
            steer = lane_steer
            lane_mode = 'warmup_lane'
        elif is_recovery:
            steer = lane_steer
            lane_mode = 'recovery_lane'
        else:
            model_correction = float(np.clip(
                model_steer, -self.model_steer_max, self.model_steer_max))
            steer = (self.lane_weight * lane_steer +
                     self.model_steer_weight * model_correction)
            lane_mode = lane_mode or 'blended'

        # Rate-limit the steer
        steer = self._rate_limit_steer(steer)

        if self.control_mode == 'full_model':
            throttle = float(np.clip(raw_throttle, 0.0, 1.0))
            brake = float(np.clip(raw_brake, 0.0, 1.0))
            return throttle, steer, brake, 'full_model', False, lane_mode, False

        # --- Throttle/brake ---
        brake = raw_brake if raw_brake >= self.brake_deadzone else 0.0
        brake = brake * self.brake_scale

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

        if brake > self.brake_conflict_threshold and not launch_protect_active:
            throttle = 0.0
            if not assist_reason:
                assist_reason = 'braking'
        elif throttle > 0.5:
            brake = min(brake, 0.05)

        is_emergency = False
        if obstacle_near and speed > self.emergency_brake_speed:
            brake = max(brake, self.emergency_brake_value)
            throttle = 0.0
            assist_reason = 'emergency'
            is_emergency = True
            steer = lane_steer

        # --- Red light / Yellow light handling ---
        if status == "Red Light":
            # Hard stop for red light
            brake = max(brake, 0.7)
            throttle = 0.0
            assist_reason = 'red_light'
        elif status == "Yellow Light":
            # Caution for yellow light
            if speed > 15.0:  # Only brake if moving reasonably fast
                brake = max(brake, 0.3)
                throttle = min(throttle, 0.2)
                assist_reason = 'yellow_light'
        elif status == "Caution":
            # Vehicle ahead too close
            if speed > 10.0:
                brake = max(brake, 0.2)
                throttle = min(throttle, 0.2)
                assist_reason = 'caution'

        # --- Speed-steer coupling: reduce throttle when turning hard ---
        abs_steer = abs(steer)
        if abs_steer > self.steer_throttle_high:
            throttle = min(throttle, 0.10)
            brake = max(brake, 0.08)
        elif abs_steer > self.steer_throttle_low:
            throttle = min(throttle, 0.18)

        # --- Off-lane safety mode ---
        is_offlane = (abs(lateral_err) > self.offlane_lat_threshold or
                      abs(heading_err) > self.offlane_hdg_threshold)
        if is_offlane:
            throttle = min(throttle, self.offlane_throttle_cap)
            brake = max(brake, self.offlane_brake_floor)
            steer = float(np.clip(steer, -self.offlane_steer_cap,
                                  self.offlane_steer_cap))
            lane_mode = 'offlane_safety'

        throttle = float(np.clip(throttle, 0.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        return throttle, steer, brake, assist_reason, is_emergency, lane_mode, is_offlane

    # ===================================================================
    #  Traffic light & status detection
    # ===================================================================
    def get_nearby_info(self, world, vehicle):
        """
        Detect nearby traffic lights and lead vehicles.
        Returns dict with: tl_state (0=Red, 1=Yellow, 2=Green), is_at_tl,
        dist_to_tl, dist_to_lead, lead_v_speed, etc.
        """
        try:
            ego_location = vehicle.get_location()
            ego_waypoint = world.get_map().get_waypoint(ego_location)
            ego_transform = vehicle.get_transform()
            ego_forward = ego_transform.get_forward_vector()
            
            # Initialize defaults
            dist_to_lead = 100.0
            lead_v_speed = 100.0
            blocked_ahead = 0
            
            # Check for lead vehicle in same lane/road
            vehicles = world.get_actors().filter('vehicle.*')
            for v in vehicles:
                if v.id == vehicle.id:
                    continue
                v_loc = v.get_location()
                dist = ego_location.distance(v_loc)
                if dist < 50.0:
                    try:
                        v_waypoint = world.get_map().get_waypoint(v_loc)
                        if (v_waypoint.road_id == ego_waypoint.road_id and 
                            v_waypoint.lane_id == ego_waypoint.lane_id):
                            # Check if ahead (dot product with ego forward)
                            ray = v_loc - ego_location
                            dot = ego_forward.x * ray.x + ego_forward.y * ray.y
                            if dot > 0:
                                v_vel = v.get_velocity()
                                v_speed = np.sqrt(v_vel.x**2 + v_vel.y**2 + v_vel.z**2) * 3.6
                                if dist < dist_to_lead:
                                    dist_to_lead = dist
                                    lead_v_speed = v_speed
                                if v_speed < 1.0 and dist < 20.0:
                                    blocked_ahead += 1
                    except Exception:
                        pass
            
            # Detect traffic light
            tl_state = 2  # Default: green
            is_at_tl = vehicle.is_at_traffic_light()
            dist_to_tl = 100.0
            
            active_tl = vehicle.get_traffic_light()
            if active_tl:
                tl_state = int(active_tl.get_state())
                dist_to_tl = ego_location.distance(active_tl.get_location())
            else:
                # Scan nearby traffic lights
                traffic_lights = world.get_actors().filter('traffic.traffic_light')
                for tl in traffic_lights:
                    tl_loc = tl.get_location()
                    d = ego_location.distance(tl_loc)
                    if d < 40.0:
                        ray = tl_loc - ego_location
                        dot = ego_forward.x * ray.x + ego_forward.y * ray.y
                        mag = np.sqrt(ray.x**2 + ray.y**2) * np.sqrt(
                            ego_forward.x**2 + ego_forward.y**2)
                        # Only consider TLs ahead with reasonable angular alignment
                        if mag > 1e-6 and (dot / mag) > 0.7 and d < dist_to_tl:
                            dist_to_tl = d
                            tl_state = int(tl.get_state())
            
            return {
                'tl_state': tl_state,           # 0=Red, 1=Yellow, 2=Green
                'is_at_tl': is_at_tl,
                'dist_to_tl': dist_to_tl,
                'dist_to_lead': dist_to_lead,
                'lead_v_speed': lead_v_speed,
                'blocked_ahead': blocked_ahead,
            }
        except Exception as e:
            # Safe defaults if detection fails
            return {
                'tl_state': 2,
                'is_at_tl': False,
                'dist_to_tl': 100.0,
                'dist_to_lead': 100.0,
                'lead_v_speed': 100.0,
                'blocked_ahead': 0,
            }

    def get_status(self, speed, nearby_info):
        """
        Determine driving status based on traffic light, vehicle ahead, etc.
        Returns status string: "Red Light", "Yellow Light", "Caution", etc.
        """
        tl_state = nearby_info.get('tl_state', 2)
        is_at_tl = nearby_info.get('is_at_tl', False)
        dist_to_tl = nearby_info.get('dist_to_tl', 100.0)
        dist_to_lead = nearby_info.get('dist_to_lead', 100.0)
        lead_v_speed = nearby_info.get('lead_v_speed', 100.0)
        
        # Red light detection
        if tl_state == 0:  # Red
            if is_at_tl or dist_to_tl < 15.0:
                return "Red Light"
            # Also treat as red if blocked ahead and TL nearby
            if (dist_to_lead < 12.0 and lead_v_speed < 2.0 and 
                dist_to_tl < 35.0):
                return "Red Light"
        
        # Yellow light detection
        if tl_state == 1:  # Yellow
            if is_at_tl or dist_to_tl < 15.0:
                return "Yellow Light"
            if (dist_to_lead < 12.0 and lead_v_speed < 5.0 and 
                dist_to_tl < 35.0):
                return "Yellow Light"
        
        # Caution: vehicle ahead too close
        if dist_to_lead < 8.0:
            return "Caution"
        
        return "Following Route"

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

        # Semi-transparent dark banner
        overlay = display.copy()
        has_lane = self.lane_assist_enabled and self.control_mode != 'full_model'
        banner_h = 390 if has_lane else (280 if self.control_mode == 'assist_throttle' else 180)
        cv2.rectangle(overlay, (0, 0), (self.screen_width, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        green = (0, 255, 0)
        yellow = (0, 255, 255)
        red = (0, 0, 255)
        cyan = (255, 255, 0)
        white = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thick = 1

        y = 22
        def put(text, color=green):
            nonlocal y
            cv2.putText(display, text, (12, y), font, scale, color, thick, cv2.LINE_AA)
            y += 24

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

        # Status display (Red Light, Yellow Light, Caution, etc.)
        status = hud_info.get('status', 'Following Route')
        status_color = green
        if status == "Red Light":
            status_color = red
        elif status == "Yellow Light":
            status_color = yellow
        elif status == "Caution":
            status_color = yellow
        put(f"Status: {status}", status_color)

        if has_lane:
            ls = hud_info.get('lane_steer', 0)
            h_err = hud_info.get('heading_err_deg', 0)
            l_err = hud_info.get('lateral_err_m', 0)
            lm = hud_info.get('lane_mode', '?')
            conf = hud_info.get('lane_confidence', 1.0)
            steer_pre = hud_info.get('steer_pre_rate', 0)
            put(f"Lane: {lm}  Conf:{conf:.1f}", cyan)
            put(f"Hdg:{h_err:+.1f}deg  Lat:{l_err:+.2f}m  LnS:{ls:+.2f}", cyan)
            put(f"Pre-RL:{steer_pre:+.2f}  Post-RL:{hud_info.get('final_steer',0):+.2f}",
                yellow)

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

            # Initialize route for this episode
            self._init_episode_route(world, vehicle, start_transform)

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

            # Lane-assist accumulators
            sum_abs_heading_err = 0.0
            sum_abs_lateral_err = 0.0
            sum_abs_lane_steer = 0.0
            lane_assist_active_ticks = 0
            self._prev_lane_steer = 0.0
            self._prev_final_steer = 0.0
            self._lane_confidence = 1.0
            self._last_stable_steer = 0.0
            offlane_ticks = 0

            # Initialize lane anchor from spawn point
            spawn_wp = world.get_map().get_waypoint(
                start_transform.location, project_to_road=True,
                lane_type=carla.LaneType.Driving)
            self._anchor_wp = spawn_wp
            if spawn_wp:
                self._anchor_road_id = spawn_wp.road_id
                self._anchor_lane_id = spawn_wp.lane_id
            else:
                self._anchor_road_id = -1
                self._anchor_lane_id = 0

            # Lane-invasion recovery state
            lane_recovery_until = 0.0
            lane_recovery_active = False
            lane_recovery_count = 0

            # Per-tick HUD state
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
                heading_err = lateral_err = 0.0
                is_junction = False
                lane_steer_val = 0.0
                lane_mode = ''
                steer_pre_rl = 0.0
                lane_active_this_tick = False

                if self.current_frame is not None:
                    frame_tensor = self.eval_transform(self.current_frame)
                    frame_buffer.append(frame_tensor)

                    if model_name in ('cnn_gru', 'gru_only') and len(frame_buffer) < 5:
                        frame_count += 1
                        speed_sum += speed
                        total_distance += float(np.sqrt(
                            velocity.x**2 + velocity.y**2 + velocity.z**2) * 0.05)
                        continue

                    with torch.no_grad():
                        speed_normalized = torch.tensor(
                            [[speed / 100.0]], dtype=torch.float32
                        )
                        if model_name in ('cnn_gru', 'gru_only'):
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
                        elapsed = time.time() - start_time

                        # --- Lane controller ---
                        extra_la = 0.0
                        if lane_recovery_active:
                            extra_la = self.lane_recovery_lookahead_bonus

                        if self.lane_assist_enabled and self.control_mode != 'full_model':
                            heading_err, lateral_err, is_junction, lane_conf = \
                                self.get_lane_errors(world, vehicle,
                                                     extra_lookahead=extra_la)
                            self._lane_confidence = lane_conf
                            lane_steer_val, lane_mode = self.compute_stanley_steer(
                                heading_err, lateral_err, speed,
                                is_junction, lane_conf)
                            lane_active_this_tick = True
                        else:
                            heading_err = lateral_err = 0.0
                            is_junction = False
                            lane_steer_val = 0.0
                            lane_mode = 'disabled'
                            lane_active_this_tick = False

                        # --- Lane-invasion recovery logic ---
                        is_recovery = False
                        if lane_recovery_active:
                            if time.time() < lane_recovery_until:
                                is_recovery = True
                            else:
                                lane_recovery_active = False
                                print(f"  [LANE] Recovery #{lane_recovery_count} ended.")
                        elif self.lane_invasion_counter > 0:
                            # New invasion detected: go to lane-only + slow down
                            lane_recovery_active = True
                            lane_recovery_count += 1
                            lane_recovery_until = (time.time() +
                                                   self.lane_invasion_recovery_sec)
                            is_recovery = True
                            print(f"  [LANE] Invasion recovery #{lane_recovery_count}: "
                                  f"lane-only for {self.lane_invasion_recovery_sec}s")

                        # Steer before rate-limiting (for HUD debug)
                        steer_pre_rl = lane_steer_val

                        # Get nearby traffic light and vehicle info
                        nearby_info = self.get_nearby_info(world, vehicle)
                        status = self.get_status(speed, nearby_info)

                        final_t, final_s, final_b, assist_reason, is_emergency, lane_mode, is_offlane = \
                            self.arbitrate_control(
                                raw_t, raw_s, raw_b, speed, obstacle_near,
                                lane_steer=lane_steer_val,
                                lane_mode=lane_mode,
                                elapsed_sec=elapsed,
                                is_recovery=is_recovery,
                                heading_err=heading_err,
                                lateral_err=lateral_err,
                                status=status
                            )

                        if is_offlane:
                            offlane_ticks += 1

                        # During recovery, reduce speed target
                        if is_recovery and speed > self.lane_recovery_speed_target:
                            final_t = min(final_t, 0.1)
                            final_b = max(final_b, 0.15)

                        # Hard launch override
                        if (self.control_mode == 'assist_throttle'
                                and elapsed < self.force_launch_duration
                                and speed < self.force_launch_speed
                                and not obstacle_near):
                            final_t = max(final_t, self.force_launch_throttle)
                            final_b = 0.0
                            final_s = float(np.clip(
                                final_s,
                                -self.force_launch_steer_clamp,
                                self.force_launch_steer_clamp))
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

                        if (final_b > 0.5 and speed < 5
                                and not obstacle_near):
                            phantom_brake_events += 1

                        sum_applied_throttle += final_t
                        sum_model_brake += float(np.clip(raw_b, 0, 1))
                        sum_applied_brake += final_b
                        if is_emergency:
                            emergency_override_count += 1
                        if self.control_mode == 'assist_throttle' and assist_reason:
                            throttle_assist_active_ticks += 1

                        if lane_active_this_tick:
                            sum_abs_heading_err += abs(heading_err)
                            sum_abs_lateral_err += abs(lateral_err)
                            sum_abs_lane_steer += abs(lane_steer_val)
                            lane_assist_active_ticks += 1

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
                    'lane_steer': lane_steer_val,
                    'heading_err_deg': math.degrees(heading_err),
                    'lateral_err_m': lateral_err,
                    'lane_mode': lane_mode,
                    'lane_confidence': self._lane_confidence,
                    'steer_pre_rate': steer_pre_rl if lane_active_this_tick else 0.0,
                    'status': status,
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

            la_ticks = max(1, lane_assist_active_ticks)

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
                'avg_abs_heading_error_deg': float(
                    math.degrees(sum_abs_heading_err / la_ticks)),
                'avg_abs_lateral_error_m': float(sum_abs_lateral_err / la_ticks),
                'avg_lane_assist_steer': float(sum_abs_lane_steer / la_ticks),
                'lane_assist_active_ratio': float(
                    lane_assist_active_ticks / infer_ticks),
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
            'avg_abs_heading_error_deg': 0.0,
            'avg_abs_lateral_error_m': 0.0,
            'avg_lane_assist_steer': 0.0,
            'lane_assist_active_ratio': 0.0,
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