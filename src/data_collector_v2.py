"""
data_collector_v2.py — Route-based, path-aware data collection for CARLA.

Run with a single command:
    python src/data_collector_v2.py

All configuration is embedded. No environment variables required.
Auto-adds CARLA PythonAPI to sys.path.
Resumes from where it left off via progress.json.
Collects 300,000 samples across Town01, Town02, Town04.
Saves EVERY frame immediately (no episode buffering).
"""

# ──────────────────────────────────────────────────────────────────────
# 1. Auto-add CARLA PythonAPI paths
# ──────────────────────────────────────────────────────────────────────
import sys
import os
import glob

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

_CARLA_CANDIDATES = [
    os.path.join(os.environ.get('CARLA_ROOT', r'C:\CARLA'), 'PythonAPI', 'carla', 'dist'),
    os.path.join(os.environ.get('CARLA_ROOT', r'C:\CARLA_0.9.15'), 'PythonAPI', 'carla', 'dist'),
    os.path.join(os.environ.get('CARLA_ROOT', r'C:\CARLA'), 'PythonAPI', 'carla'),
    os.path.join(os.environ.get('CARLA_ROOT', r'C:\CARLA_0.9.15'), 'PythonAPI', 'carla'),
    os.path.join(os.environ.get('CARLA_ROOT', r'C:\CARLA'), 'PythonAPI'),
    os.path.join(os.environ.get('CARLA_ROOT', r'C:\CARLA_0.9.15'), 'PythonAPI'),
]

for _candidate in _CARLA_CANDIDATES:
    if os.path.isdir(_candidate):
        for _egg in glob.glob(os.path.join(_candidate, '*.egg')):
            if _egg not in sys.path:
                sys.path.insert(0, _egg)
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ──────────────────────────────────────────────────────────────────────
import carla
import cv2
import numpy as np
import csv
import time
import math
import random
import json
import re
import traceback
from collections import Counter
from PIL import Image

from path_features import compute_path_features, heading_delta

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False
    print("WARNING: GlobalRoutePlanner not available. Route-based collection disabled.")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
TOTAL_TARGET_SAMPLES      = 300_000
TRAINING_TOWNS            = ['Town01', 'Town02', 'Town04']
BG_VEHICLE_COUNT          = 60
BG_WALKER_COUNT           = 40
STUCK_TICK_THRESHOLD      = 400     # ~40 seconds at 10Hz

WEATHER_PRESETS = [
    ('ClearNoon',   carla.WeatherParameters.ClearNoon),
    ('WetNoon',     carla.WeatherParameters.WetNoon),
    ('CloudyNoon',  carla.WeatherParameters.CloudyNoon),
    ('ClearSunset', carla.WeatherParameters.ClearSunset),
]

TRAFFIC_SEEDS = [42, 137, 256]

EGO_BLUEPRINTS = [
    'vehicle.tesla.model3', 'vehicle.audi.tt',
    'vehicle.bmw.grandtourer', 'vehicle.toyota.prius',
]
BG_BLUEPRINTS = [
    'vehicle.tesla.model3', 'vehicle.audi.tt', 'vehicle.audi.a2',
    'vehicle.toyota.prius', 'vehicle.citroen.c3', 'vehicle.nissan.micra',
]

# ── CSV schema (27 columns) ──────────────────────────────────────────
CSV_COLUMNS = [
    'timestamp', 'frame_id', 'speed', 'speed_limit',
    'throttle', 'brake', 'steer',
    'gap', 'status', 'lane_id', 'road_id', 'weather', 'town',
    'hdg_delta_1', 'hdg_delta_2', 'hdg_delta_3', 'hdg_delta_4', 'hdg_delta_5',
    'curvature_near', 'curvature_mid', 'curvature_far',
    'dist_to_junction', 'turn_intent', 'route_progress',
    'tl_class', 'stop_required', 'steer_smooth',
]

CSV_DEFAULTS = {
    'timestamp': 0.0, 'frame_id': '0000000.jpg', 'speed': 0.0,
    'speed_limit': 30.0, 'throttle': 0.0, 'brake': 0.0, 'steer': 0.0,
    'gap': 100.0, 'status': 'Unknown', 'lane_id': 0, 'road_id': 0,
    'weather': 'Unknown', 'town': 'Unknown',
    'hdg_delta_1': 0.0, 'hdg_delta_2': 0.0, 'hdg_delta_3': 0.0,
    'hdg_delta_4': 0.0, 'hdg_delta_5': 0.0,
    'curvature_near': 0.0, 'curvature_mid': 0.0, 'curvature_far': 0.0,
    'dist_to_junction': 100.0, 'turn_intent': 'straight',
    'route_progress': 0.0, 'tl_class': 'none', 'stop_required': 0,
    'steer_smooth': 0.0,
}


# ══════════════════════════════════════════════════════════════════════
class CarlaDataCollectorV2:

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)

        # ── Paths ──
        self.dataset_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, 'dataset_v2'))
        self.images_dir  = os.path.normpath(os.path.join(self.dataset_dir, 'images'))
        os.makedirs(self.images_dir, exist_ok=True)

        self.csv_file_path = os.path.join(self.dataset_dir, 'log.csv')
        self.progress_path = os.path.join(self.dataset_dir, 'progress.json')

        # ── State (set before _load_progress) ──
        self.total_frames      = 0
        self.collected_by_town = {t: 0 for t in TRAINING_TOWNS}
        self._next_frame_id    = 0
        self._current_town     = None
        self._current_weather  = None
        self._current_seed     = None

        # ── Restore progress ──
        self._load_progress()

        # ── CSV ──
        self._init_csv()

        # ── Integrity check ──
        self._startup_integrity_check()

        # ── Live state ──
        self._steer_ema      = 0.0
        self.collision_event = False
        self.current_frame   = None
        self.vis_frame       = None
        self.stuck_ticks     = 0
        self.recovery_mode   = False
        self.recovery_timer  = 0

        # ── Counters ──
        self._intent_counts  = Counter()
        self._tl_counts      = Counter()
        self._status_counts  = Counter()

        print(f"\n{'='*60}")
        print(f"  CARLA Path-Aware Data Collector v2")
        print(f"  Target: {TOTAL_TARGET_SAMPLES:,} samples")
        print(f"  Dataset: {self.dataset_dir}")
        print(f"  Total collected: {self.total_frames:,}")
        print(f"  Per-town: {dict(self.collected_by_town)}")
        print(f"  Next frame ID: {self._next_frame_id}")
        print(f"  NPCs: {BG_VEHICLE_COUNT} vehicles, {BG_WALKER_COUNT} walkers")
        print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════════════════════
    #  Progress persistence
    # ══════════════════════════════════════════════════════════════════
    def _load_progress(self):
        if os.path.isfile(self.progress_path):
            try:
                with open(self.progress_path, 'r') as f:
                    prog = json.load(f)
                self.total_frames      = prog.get('total_samples', 0)
                self.collected_by_town = prog.get('per_town', {t: 0 for t in TRAINING_TOWNS})
                self._next_frame_id    = prog.get('next_frame_id', self.total_frames)
                self._current_town     = prog.get('current_town', None)
                self._current_weather  = prog.get('current_weather', None)
                self._current_seed     = prog.get('current_seed', None)
                print(f"Resumed from progress.json — {self.total_frames:,} samples")
                return
            except Exception as e:
                print(f"Warning: Could not read progress.json ({e}), scanning CSV...")

        # Fallback: scan CSV
        if os.path.isfile(self.csv_file_path):
            try:
                max_id = -1
                with open(self.csv_file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.total_frames += 1
                        town = (row.get('town') or '').strip()
                        if town in self.collected_by_town:
                            self.collected_by_town[town] += 1
                        fid = row.get('frame_id', '')
                        match = re.match(r'^(\d+)\.(jpg|png)$', fid, re.IGNORECASE)
                        if match:
                            num = int(match.group(1))
                            if num > max_id:
                                max_id = num
                    self._next_frame_id = max_id + 1 if max_id >= 0 else self.total_frames
            except Exception:
                self._next_frame_id = 0
        print(f"Scanned CSV — {self.total_frames:,} samples, next ID: {self._next_frame_id}")

    def _save_progress(self):
        try:
            with open(self.progress_path, 'w') as f:
                json.dump({
                    'total_samples':   self.total_frames,
                    'per_town':        self.collected_by_town,
                    'next_frame_id':   self._next_frame_id,
                    'current_town':    self._current_town,
                    'current_weather': self._current_weather,
                    'current_seed':    self._current_seed,
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress.json: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  CSV management
    # ══════════════════════════════════════════════════════════════════
    def _init_csv(self):
        file_exists = os.path.isfile(self.csv_file_path) and os.path.getsize(self.csv_file_path) > 0
        self.csv_file   = open(self.csv_file_path, 'a', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(CSV_COLUMNS)
            self.csv_file.flush()

    def _write_row(self, row_dict):
        """Write one row, filling defaults for any missing field."""
        row = []
        for col in CSV_COLUMNS:
            val = row_dict.get(col)
            if val is None or (isinstance(val, str) and val.strip() == ''):
                val = CSV_DEFAULTS[col]
            row.append(val)
        self.csv_writer.writerow(row)
        return row

    # ══════════════════════════════════════════════════════════════════
    #  Startup integrity check
    # ══════════════════════════════════════════════════════════════════
    def _startup_integrity_check(self):
        csv_rows = 0
        if os.path.isfile(self.csv_file_path):
            try:
                with open(self.csv_file_path, 'r') as f:
                    for _ in csv.reader(f):
                        csv_rows += 1
                csv_rows = max(0, csv_rows - 1)  # subtract header
            except Exception:
                pass

        img_count = 0
        if os.path.isdir(self.images_dir):
            img_count = len([f for f in os.listdir(self.images_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"  Integrity: {csv_rows:,} CSV rows, {img_count:,} images", end="")
        if csv_rows != img_count:
            print(f"  ⚠ MISMATCH")
        else:
            print(f"  ✓ OK")

    # ══════════════════════════════════════════════════════════════════
    #  Route planning
    # ══════════════════════════════════════════════════════════════════
    def _plan_route(self, world, spawn_transform):
        if not HAS_PLANNER:
            return [], None
        carla_map = world.get_map()
        try:
            planner = GlobalRoutePlanner(carla_map, 2.0)
        except Exception as e:
            print(f"    Planner init failed: {e}")
            return [], None

        spawn_loc = spawn_transform.location
        spawn_wp  = carla_map.get_waypoint(spawn_loc, project_to_road=True)
        if spawn_wp is None:
            return [], None

        dest_wp = spawn_wp
        target_dist = random.uniform(300, 600)
        walked = 0.0
        while walked < target_dist:
            nexts = dest_wp.next(5.0)
            if not nexts:
                break
            dest_wp = random.choice(nexts)
            walked += 5.0

        try:
            route = planner.trace_route(spawn_loc, dest_wp.transform.location)
            return route, planner
        except Exception as e:
            print(f"    Route planning failed: {e}")
            return [], None

    # ══════════════════════════════════════════════════════════════════
    #  Nearby info
    # ══════════════════════════════════════════════════════════════════
    def _get_nearby_info(self, world, vehicle):
        ego_loc = vehicle.get_location()
        ego_wp  = world.get_map().get_waypoint(ego_loc)
        ego_fwd = vehicle.get_transform().get_forward_vector()

        dist_to_lead  = 100.0
        lead_v_speed  = 100.0
        blocked_ahead = 0

        for v in world.get_actors().filter('vehicle.*'):
            if v.id == vehicle.id:
                continue
            v_loc = v.get_location()
            dist  = ego_loc.distance(v_loc)
            if dist < 50.0:
                try:
                    v_wp = world.get_map().get_waypoint(v_loc)
                    if v_wp.road_id == ego_wp.road_id and v_wp.lane_id == ego_wp.lane_id:
                        ray = v_loc - ego_loc
                        if (ego_fwd.x * ray.x + ego_fwd.y * ray.y) > 0:
                            v_vel   = v.get_velocity()
                            v_speed = np.sqrt(v_vel.x**2 + v_vel.y**2 + v_vel.z**2) * 3.6
                            if dist < dist_to_lead:
                                dist_to_lead = dist
                                lead_v_speed = v_speed
                            if v_speed < 1.0 and dist < 20.0:
                                blocked_ahead += 1
                except Exception:
                    pass

        # Traffic light
        tl_state   = 2
        is_at_tl   = vehicle.is_at_traffic_light()
        dist_to_tl = 100.0
        active_tl  = vehicle.get_traffic_light()
        if active_tl:
            tl_state   = int(active_tl.get_state())
            dist_to_tl = ego_loc.distance(active_tl.get_location())
        else:
            # Extended TL search (from v1)
            for tl in world.get_actors().filter('traffic.traffic_light'):
                tl_loc = tl.get_location()
                d = ego_loc.distance(tl_loc)
                if d < 40.0:
                    ray = tl_loc - ego_loc
                    if (ego_fwd.x * ray.x + ego_fwd.y * ray.y) > 0:
                        mag = np.sqrt(ray.x**2 + ray.y**2) * np.sqrt(ego_fwd.x**2 + ego_fwd.y**2)
                        dot = (ego_fwd.x * ray.x + ego_fwd.y * ray.y)
                        if dot / (mag + 1e-6) > 0.7 and d < dist_to_tl:
                            dist_to_tl = d
                            tl_state   = int(tl.get_state())

        return {
            'tl_state': tl_state, 'is_at_tl': is_at_tl,
            'dist_to_tl': dist_to_tl,
            'dist_to_lead': dist_to_lead, 'lead_v_speed': lead_v_speed,
            'blocked_ahead': blocked_ahead,
            'speed_limit': vehicle.get_speed_limit(),
            'lane_id': ego_wp.lane_id, 'road_id': ego_wp.road_id,
        }

    def _get_status(self, speed, control, nearby):
        if self.collision_event:
            self.collision_event = False
            self.recovery_mode  = True
            self.recovery_timer = 30
            return 'Emergency Stop'

        if self.recovery_mode:
            self.recovery_timer -= 1
            if self.recovery_timer <= 0:
                self.recovery_mode = False
                return 'Recovered'
            return 'Caution'

        if nearby['tl_state'] == 0:
            if nearby['is_at_tl'] or nearby['dist_to_tl'] < 15.0:
                return 'Red Light'
            if nearby['dist_to_lead'] < 12.0 and nearby['lead_v_speed'] < 2.0:
                return 'Red Light'
        if nearby['tl_state'] == 1:
            if nearby['is_at_tl'] or nearby['dist_to_tl'] < 15.0:
                return 'Yellow Light'
        if nearby['dist_to_lead'] < 8.0:
            return 'Caution'
        if speed < 0.5 and control.brake > 0.5:
            return 'Stopped'
        return 'Following Route'

    def _advance_route_idx(self, ego_loc, route, idx):
        if idx >= len(route):
            return idx
        wp, _ = route[idx]
        if ego_loc.distance(wp.transform.location) < 3.0:
            return idx + 1
        return idx

    # ══════════════════════════════════════════════════════════════════
    #  Spawning (v1-style, proven to work)
    # ══════════════════════════════════════════════════════════════════
    def _is_spawn_clear(self, world, location):
        for actor in world.get_actors().filter('vehicle.*'):
            if actor.get_location().distance(location) < 10.0:
                return False
        return True

    def _spawn_ego(self, world, tm):
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.find(random.choice(EGO_BLUEPRINTS))
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for sp in spawn_points:
            if self._is_spawn_clear(world, sp.location):
                vehicle = world.try_spawn_actor(bp, sp)
                if vehicle:
                    vehicle.set_autopilot(True, tm.get_port())
                    tm.auto_lane_change(vehicle, True)
                    tm.distance_to_leading_vehicle(vehicle, 2.0)
                    tm.vehicle_percentage_speed_difference(vehicle, -10.0)
                    return vehicle, sp
        return None, None

    def _spawn_traffic(self, world, tm, seed):
        random.seed(seed)
        bp_lib = world.get_blueprint_library()
        spawns = world.get_map().get_spawn_points()
        random.shuffle(spawns)
        actors = []

        # ── Vehicles ──
        spawned = 0
        for sp in spawns:
            if spawned >= BG_VEHICLE_COUNT:
                break
            if self._is_spawn_clear(world, sp.location):
                bp = bp_lib.find(random.choice(BG_BLUEPRINTS))
                actor = world.try_spawn_actor(bp, sp)
                if actor:
                    actor.set_autopilot(True, tm.get_port())
                    tm.auto_lane_change(actor, False)
                    tm.distance_to_leading_vehicle(actor, 5.0)
                    tm.vehicle_percentage_speed_difference(actor, random.uniform(8.0, 20.0))
                    tm.ignore_lights_percentage(actor, 0)
                    try:
                        tm.random_left_lanechange_percentage(actor, 0)
                        tm.random_right_lanechange_percentage(actor, 0)
                    except Exception:
                        pass
                    actors.append(actor)
                    spawned += 1

        # ── Walkers ──
        walker_actors = []
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        spawned_walkers = 0
        if walker_bps:
            for _ in range(BG_WALKER_COUNT * 2):
                if spawned_walkers >= BG_WALKER_COUNT:
                    break
                loc = world.get_random_location_from_navigation()
                if loc is None:
                    continue
                sp = carla.Transform()
                sp.location = loc
                wbp = random.choice(walker_bps)
                if wbp.has_attribute('is_invincible'):
                    wbp.set_attribute('is_invincible', 'false')
                walker = world.try_spawn_actor(wbp, sp)
                if walker:
                    walker_actors.append(walker)
                    spawned_walkers += 1

        print(f"    Spawned {spawned} vehicles, {spawned_walkers} walkers")
        return actors, walker_actors

    # ══════════════════════════════════════════════════════════════════
    #  Camera
    # ══════════════════════════════════════════════════════════════════
    def _setup_camera(self, world, vehicle):
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '640')
        bp.set_attribute('image_size_y', '480')
        camera = world.spawn_actor(
            bp, carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=vehicle)
        camera.listen(self._process_image)
        return camera

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.vis_frame     = array.copy()
        self.current_frame = Image.fromarray(cv2.resize(array, (224, 224)))

    # ══════════════════════════════════════════════════════════════════
    #  Stuck / pileup check (from v1)
    # ══════════════════════════════════════════════════════════════════
    def _check_stuck(self, speed, nearby):
        # Don't count as stuck at red lights
        if nearby['tl_state'] == 0:
            self.stuck_ticks = 0
            return False

        if (nearby['blocked_ahead'] >= 1 and speed < 2.0) or speed < 0.2:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        if self.stuck_ticks > STUCK_TICK_THRESHOLD:
            print("    Stuck/Pileup detected → ending session for strategic respawn")
            return True
        return False

    # ══════════════════════════════════════════════════════════════════
    #  HUD
    # ══════════════════════════════════════════════════════════════════
    def _draw_hud(self, town_name, weather_name, speed, status,
                  control, nearby, path_feats, steer_smooth):
        if self.vis_frame is None:
            return

        disp = cv2.cvtColor(self.vis_frame, cv2.COLOR_RGB2BGR)
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (380, 460), (0, 0, 0), -1)
        disp = cv2.addWeighted(overlay, 0.60, disp, 0.40, 0)

        y = 18
        def put(txt, color=(255, 255, 255), scale=0.42):
            nonlocal y
            cv2.putText(disp, txt, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                        cv2.LINE_AA)
            y += 17

        put(f"{town_name} | {weather_name}", (0, 255, 255), 0.48)
        y += 2
        put(f"Speed: {speed:.1f} km/h   Limit: {nearby['speed_limit']:.0f}")
        put(f"Thr: {control.throttle:.2f}  Brk: {control.brake:.2f}  "
            f"Str: {control.steer:.3f}")
        put(f"SteerSmooth: {steer_smooth:.4f}")

        gap = nearby['dist_to_lead']
        sc = (0, 255, 0) if status == 'Following Route' else (0, 165, 255)
        put(f"Gap: {gap:.1f}m   Status: {status}", sc)
        put(f"Lane: {nearby['lane_id']}   Road: {nearby['road_id']}")

        y += 2
        hdg = "  ".join(f"{path_feats.get(f'hdg_delta_{i}', 0):.2f}" for i in range(1, 6))
        put(f"HdgD: {hdg}", (200, 200, 255))
        put(f"Curv: N={path_feats.get('curvature_near',0):.4f}  "
            f"M={path_feats.get('curvature_mid',0):.4f}  "
            f"F={path_feats.get('curvature_far',0):.4f}")
        put(f"Junc: {path_feats.get('dist_to_junction',100):.0f}m  "
            f"Intent: {path_feats.get('turn_intent','straight')}",
            (255, 200, 100))
        rp = path_feats.get('route_progress', 0)
        put(f"Route: {rp:.0%}   TL: {path_feats.get('tl_class','none')}  "
            f"Stop: {int(path_feats.get('stop_required', 0))}")

        y += 4
        put(f"TOTAL: {self.total_frames:,} / {TOTAL_TARGET_SAMPLES:,}",
            (0, 255, 0), 0.50)

        # Progress bar
        pct = min(1.0, self.total_frames / TOTAL_TARGET_SAMPLES)
        bw, bh = 360, 12
        bx, by = 8, y + 2
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
        cv2.rectangle(disp, (bx, by), (bx + int(bw * pct), by + bh), (0, 255, 100), -1)

        cv2.imshow("CARLA Data Collector v2", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n  User pressed Q — saving and exiting.")
            self._save_progress()
            self.csv_file.flush()
            cv2.destroyAllWindows()
            sys.exit(0)

    # ══════════════════════════════════════════════════════════════════
    #  Quality gate (periodic report only, does NOT discard data)
    # ══════════════════════════════════════════════════════════════════
    def _quality_report(self):
        total = sum(self._intent_counts.values())
        if total == 0:
            return
        print(f"\n{'─'*50}")
        print(f"  Quality Report @ {self.total_frames:,} frames")
        print(f"  Intents: {dict(self._intent_counts)}")
        print(f"  TL:      {dict(self._tl_counts)}")
        print(f"  Status:  {dict(self._status_counts)}")
        print(f"{'─'*50}\n")

    # ══════════════════════════════════════════════════════════════════
    #  Main collection loop
    # ══════════════════════════════════════════════════════════════════
    def collect(self):
        print("Starting Path-Aware Data Collection v2...\n")

        # Reset sync mode
        try:
            world = self.client.get_world()
            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
        except Exception:
            pass

        self.client.set_timeout(120.0)

        if self.total_frames >= TOTAL_TARGET_SAMPLES:
            print(f"Already at {self.total_frames:,}. Done.")
            self._print_final_summary()
            return

        town_target = TOTAL_TARGET_SAMPLES // len(TRAINING_TOWNS)

        for town_name in TRAINING_TOWNS:
            if self.total_frames >= TOTAL_TARGET_SAMPLES:
                break

            town_frames = self.collected_by_town.get(town_name, 0)
            if town_frames >= town_target:
                print(f"\n{town_name} complete ({town_frames:,}). Skipping.")
                continue

            self._current_town = town_name
            print(f"\n{'═'*60}")
            print(f"  Loading {town_name}  ({town_frames:,} / {town_target:,})")
            print(f"{'═'*60}")

            # ── Robust Map Load (3 retries) ──
            success = False
            for attempt in range(3):
                try:
                    print(f"  Attempting to load {town_name} (Attempt {attempt+1}/3)...")
                    self.client.load_world(town_name)
                    time.sleep(5.0)
                    success = True
                    break
                except Exception as e:
                    print(f"  ✗ Load failed: {e}. Retrying in 10s...")
                    time.sleep(10.0)
            
            if not success:
                print(f"  ✗ Skipping {town_name} due to repeated load failures.")
                continue

            world = self.client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            world.apply_settings(settings)

            tm = self.client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            tm.set_global_distance_to_leading_vehicle(4.0)
            try:
                tm.set_hybrid_physics_mode(False)
            except Exception:
                pass

            for weather_name, weather_preset in WEATHER_PRESETS:
                if self.total_frames >= TOTAL_TARGET_SAMPLES:
                    break
                if town_frames >= town_target:
                    break

                world.set_weather(weather_preset)
                self._current_weather = weather_name
                print(f"\n  Weather: {weather_name}")

                for seed in TRAFFIC_SEEDS:
                    if self.total_frames >= TOTAL_TARGET_SAMPLES:
                        break
                    if town_frames >= town_target:
                        break

                    self._current_seed = seed
                    print(f"  Seed: {seed}")

                    # ── Robust Traffic Spawn ──
                    traffic_actors, walker_actors = [], []
                    try:
                        traffic_actors, walker_actors = self._spawn_traffic(world, tm, seed)
                    except Exception as e:
                        print(f"    ✗ Traffic spawn failed: {e}. Trying to continue...")
                    
                    # Multiple sessions per seed
                    sessions = 0
                    while (town_frames < town_target
                           and self.total_frames < TOTAL_TARGET_SAMPLES
                           and sessions < 40): # Increased session limit per seed
                        sessions += 1

                        try:
                            collected = self._run_session(
                                world, tm, town_name, weather_name)
                        except Exception as e:
                            print(f"    ✗ Session error: {e}")
                            traceback.print_exc()
                            collected = 0
                            time.sleep(1.0)

                        town_frames += collected
                        self.collected_by_town[town_name] = town_frames
                        self._save_progress()

                        if self.total_frames % 20000 < 500 and self.total_frames > 0:
                            self._quality_report()

                    # Cleanup
                    for a in traffic_actors + walker_actors:
                        try:
                            if a.is_alive:
                                a.destroy()
                        except Exception:
                            pass

            # Restore async
            try:
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except Exception:
                pass

        # Finalize
        self.csv_file.flush()
        self.csv_file.close()
        self._save_progress()
        cv2.destroyAllWindows()
        self._print_final_summary()

    # ══════════════════════════════════════════════════════════════════
    #  Single driving session — SAVES EVERY FRAME IMMEDIATELY
    # ══════════════════════════════════════════════════════════════════
    def _run_session(self, world, tm, town_name, weather_name):
        vehicle          = None
        camera           = None
        collision_sensor = None
        collected        = 0
        self._steer_ema  = 0.0
        self.collision_event = False
        self.recovery_mode   = False
        self.stuck_ticks     = 0

        try:
            vehicle, spawn_tf = self._spawn_ego(world, tm)
            if vehicle is None:
                print("    ✗ Could not spawn ego")
                return 0

            camera = self._setup_camera(world, vehicle)

            # Collision sensor
            col_bp = world.get_blueprint_library().find('sensor.other.collision')
            collision_sensor = world.spawn_actor(
                col_bp, carla.Transform(), attach_to=vehicle)
            collision_sensor.listen(lambda e: setattr(self, 'collision_event', True))

            # Plan route
            route, _ = self._plan_route(world, spawn_tf)
            route_idx = 0

            # Anchor waypoint
            anchor_wp = world.get_map().get_waypoint(
                spawn_tf.location, project_to_road=True,
                lane_type=carla.LaneType.Driving)

            # Warm up (5 ticks)
            for _ in range(5):
                world.tick()

            spectator = world.get_spectator()
            print(f"    Session started: route={len(route)} wps")

            max_ticks = 4000  # ~400 seconds
            for tick in range(max_ticks):
                if self.total_frames >= TOTAL_TARGET_SAMPLES:
                    break

                try:
                    world.tick()
                except Exception:
                    continue

                if not vehicle.is_alive:
                    break

                velocity = vehicle.get_velocity()
                speed = float(np.sqrt(
                    velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6)
                control = vehicle.get_control()
                ego_t   = vehicle.get_transform()
                ego_loc = ego_t.location
                ego_fwd = ego_t.get_forward_vector()

                # Spectator follow
                try:
                    spectator.set_transform(carla.Transform(
                        ego_t.location + ego_fwd * -10 + carla.Location(z=5),
                        carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw)))
                except Exception:
                    pass

                # Advance route
                if route:
                    route_idx = self._advance_route_idx(ego_loc, route, route_idx)
                    if route_idx >= len(route):
                        print(f"    Route complete at tick {tick}")
                        break

                # Update anchor
                if anchor_wp:
                    anc_fwd = anchor_wp.transform.get_forward_vector()
                    dx = ego_loc.x - anchor_wp.transform.location.x
                    dy = ego_loc.y - anchor_wp.transform.location.y
                    along = dx * anc_fwd.x + dy * anc_fwd.y
                    if along > 1.5:
                        nexts = anchor_wp.next(min(along + 1.0, 6.0))
                        if nexts:
                            anchor_wp = nexts[0]

                # Nearby info
                try:
                    nearby = self._get_nearby_info(world, vehicle)
                except Exception:
                    nearby = {
                        'tl_state': 2, 'is_at_tl': False, 'dist_to_tl': 100.0,
                        'dist_to_lead': 100.0, 'lead_v_speed': 100.0,
                        'blocked_ahead': 0, 'speed_limit': 30.0,
                        'lane_id': 0, 'road_id': 0,
                    }

                status = self._get_status(speed, control, nearby)

                # Stuck check
                if self._check_stuck(speed, nearby):
                    break

                # Path features
                try:
                    path_feats = compute_path_features(
                        vehicle, ego_fwd, anchor_wp,
                        route_wps_with_cmds=(route if route else None),
                        route_idx=route_idx,
                        route_total=max(1, len(route)))
                except Exception:
                    path_feats = {
                        'hdg_delta_1': 0.0, 'hdg_delta_2': 0.0,
                        'hdg_delta_3': 0.0, 'hdg_delta_4': 0.0,
                        'hdg_delta_5': 0.0,
                        'curvature_near': 0.0, 'curvature_mid': 0.0,
                        'curvature_far': 0.0,
                        'dist_to_junction': 100.0, 'turn_intent': 'straight',
                        'route_progress': 0.0, 'tl_class': 'none',
                        'stop_required': False,
                    }

                # Smooth steer (EMA α=0.85)
                self._steer_ema = 0.85 * self._steer_ema + 0.15 * control.steer
                steer_smooth = round(self._steer_ema, 4)

                # ── SAVE IMMEDIATELY (image first, then CSV) ──
                if self.current_frame is not None:
                    frame_name = f"{self._next_frame_id:07d}.jpg"

                    # 1. Save image
                    self.current_frame.save(
                        os.path.join(self.images_dir, frame_name))

                    # 2. Write CSV row
                    timestamp = world.get_snapshot().timestamp.elapsed_seconds
                    self._write_row({
                        'timestamp':       round(timestamp, 4),
                        'frame_id':        frame_name,
                        'speed':           round(speed, 2),
                        'speed_limit':     nearby['speed_limit'],
                        'throttle':        round(control.throttle, 3),
                        'brake':           round(control.brake, 3),
                        'steer':           round(control.steer, 4),
                        'gap':             round(nearby['dist_to_lead'], 1),
                        'status':          status,
                        'lane_id':         nearby['lane_id'],
                        'road_id':         nearby['road_id'],
                        'weather':         weather_name,
                        'town':            town_name,
                        'hdg_delta_1':     path_feats.get('hdg_delta_1', 0.0),
                        'hdg_delta_2':     path_feats.get('hdg_delta_2', 0.0),
                        'hdg_delta_3':     path_feats.get('hdg_delta_3', 0.0),
                        'hdg_delta_4':     path_feats.get('hdg_delta_4', 0.0),
                        'hdg_delta_5':     path_feats.get('hdg_delta_5', 0.0),
                        'curvature_near':  path_feats.get('curvature_near', 0.0),
                        'curvature_mid':   path_feats.get('curvature_mid', 0.0),
                        'curvature_far':   path_feats.get('curvature_far', 0.0),
                        'dist_to_junction': path_feats.get('dist_to_junction', 100.0),
                        'turn_intent':     path_feats.get('turn_intent', 'straight'),
                        'route_progress':  path_feats.get('route_progress', 0.0),
                        'tl_class':        path_feats.get('tl_class', 'none'),
                        'stop_required':   int(path_feats.get('stop_required', 0)),
                        'steer_smooth':    steer_smooth,
                    })
                    self.csv_file.flush()

                    # 3. Increment counters
                    self._next_frame_id += 1
                    self.total_frames   += 1
                    collected           += 1

                    # Track distributions
                    intent = path_feats.get('turn_intent', 'straight')
                    tl     = path_feats.get('tl_class', 'none')
                    self._intent_counts[intent] += 1
                    self._tl_counts[tl]         += 1
                    self._status_counts[status] += 1

                # HUD (every 3rd tick to reduce overhead)
                if tick % 3 == 0:
                    self._draw_hud(town_name, weather_name, speed, status,
                                   control, nearby, path_feats, steer_smooth)

            print(f"    ✓ Session done: {collected} frames  "
                  f"(total: {self.total_frames:,})")
            return collected

        except Exception as e:
            print(f"    ✗ Session error: {e}")
            traceback.print_exc()
            return collected  # return whatever we saved so far

        finally:
            if camera and camera.is_alive:
                try:
                    camera.stop()
                    camera.destroy()
                except Exception:
                    pass
            if collision_sensor and collision_sensor.is_alive:
                try:
                    collision_sensor.stop()
                    collision_sensor.destroy()
                except Exception:
                    pass
            if vehicle and vehicle.is_alive:
                try:
                    vehicle.destroy()
                except Exception:
                    pass
            self.current_frame = None
            self.vis_frame     = None

    # ══════════════════════════════════════════════════════════════════
    #  Final summary
    # ══════════════════════════════════════════════════════════════════
    def _print_final_summary(self):
        img_count = 0
        if os.path.isdir(self.images_dir):
            img_count = len([f for f in os.listdir(self.images_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        csv_rows = 0
        if os.path.isfile(self.csv_file_path):
            try:
                with open(self.csv_file_path, 'r') as f:
                    for _ in csv.reader(f):
                        csv_rows += 1
                csv_rows = max(0, csv_rows - 1)
            except Exception:
                pass

        print(f"\n{'═'*60}")
        print(f"  COLLECTION COMPLETE")
        print(f"{'═'*60}")
        print(f"  Total CSV rows:    {csv_rows:,}")
        print(f"  Total images:      {img_count:,}")
        print(f"  Missing rows/imgs: {abs(csv_rows - img_count)}")
        print()
        print(f"  Per-town:")
        for t in TRAINING_TOWNS:
            print(f"    {t}: {self.collected_by_town.get(t, 0):,}")
        print()
        if self._status_counts:
            print(f"  Status distribution:")
            for s, c in sorted(self._status_counts.items(), key=lambda x: -x[1]):
                print(f"    {s}: {c:,}")
        if self._intent_counts:
            print(f"  Turn intent distribution:")
            for i, c in sorted(self._intent_counts.items(), key=lambda x: -x[1]):
                print(f"    {i}: {c:,}")
        if self._tl_counts:
            print(f"  Traffic light distribution:")
            for t, c in sorted(self._tl_counts.items(), key=lambda x: -x[1]):
                print(f"    {t}: {c:,}")
        print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    collector = CarlaDataCollectorV2()
    collector.collect()
