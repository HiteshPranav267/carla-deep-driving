import carla
import cv2
import pygame
import numpy as np
import torch
import time
import os
import csv
import math
import sys
from torchvision import transforms
from PIL import Image
from model import EndToEndDrivingModel

class GTAPilot:
    """ 
    Carla Deep Driving Autonomous Agent:
    - Phase 1: Shows a top-down MAP of the road network.
    - User CLICKS on a road to set a destination (green marker).
    - User presses ENTER to start the ride.
    - Phase 2: Car drives to the destination using Neural Network + Safety.
    """
    
    MAP_SIZE = 800  # Pygame window for map phase
    WIN_W, WIN_H = 1280, 720  # Driving phase window (resizable)
    CAM_W, CAM_H = 1280, 720

    def __init__(self, model_path="models/best_model.pth"):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((self.MAP_SIZE, self.MAP_SIZE))
        pygame.display.set_caption("Carla Deep Driving")
        self.font = pygame.font.SysFont("Arial", 18, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 28, bold=True)

        # Telemetry log
        self.log_path = "logs.csv"
        self.log_file = open(self.log_path, mode='w', newline='', encoding='utf-8')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["Timestamp", "Speed_KMH", "Throttle", "Brake", "Steer", "Gap", "Status"])

        # CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.tm = self.client.get_trafficmanager(8000)
        
        # Available maps for the dropdown
        self.available_maps = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"]
        self.current_map_name = self.carla_map.name.split('/')[-1]
        self.map_dropdown_open = False

        # --- Smart Pathfinding Setup ---
        # Add CARLA agents to path
        carla_root = "C:/Users/hites/CarlaUE5/PythonAPI/carla"
        if carla_root not in sys.path:
            sys.path.append(carla_root)
        # God Mode Settings
        self.settings = {'weather': 'Clear', 'vehicles': 0, 'pedestrians': 0}
        self.weather_options = ['Clear', 'Rain', 'Night', 'Fog']
        self.traffic_options = [0, 20, 50, 100]
        self.pedestrian_options = [0, 20, 50, 100]
        self.spawned_actors = []

        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            self.planner = GlobalRoutePlanner(self.carla_map, 2.0)
            print("✅ Global Route Planner initialized.")
        except ImportError:
            self.planner = None
            print("⚠️ Could not import GlobalRoutePlanner. Pathfinding will be limited.")

        # Neural Network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EndToEndDrivingModel().to(self.device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                print("Neural Network Model Loaded.")
            except:
                print("Could not load model weights. Using random weights.")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # State
        self.vehicle = None
        self.camera = None
        self.obstacle_sensor = None
        self.collision_sensor = None
        self.current_frame = None
        self.obstacle_dist = 50.0
        self.destination_wp = None
        self.spawn_wp = None
        self.waypoints = []
        self.route = []               # Planned route (world coords)
        self.breadcrumb_trail = []    # Actual path taken (world coords)
        self.last_collision_time = 0
        self.stuck_timer = 0
        self.recovering = False
        self.recover_start = 0
        self._breadcrumb_counter = 0
        self.current_tm_path = []  # Store current active route for TM
        
        # Map rendering data
        self.map_surface = None
        self.minimap_surface = None   # 250x250 version for driving HUD
        self.map_bounds = None

        # Speed/safety
        self.target_speed = 30.0
        self.smoothing_alpha = 0.25
        self.s_throttle = 0.0
        self.s_brake = 0.0

        # Build map data
        self._build_map_data()

    def _build_map_data(self):
        """Pre-compute all road waypoints for the top-down map."""
        print("Building road network map...")
        self.waypoints = self.carla_map.generate_waypoints(5.0)
        
        xs = [wp.transform.location.x for wp in self.waypoints]
        ys = [wp.transform.location.y for wp in self.waypoints]
        
        # Also include official spawn points in bounds so car is never off-map
        spawn_points = self.carla_map.get_spawn_points()
        for sp in spawn_points:
            xs.append(sp.location.x)
            ys.append(sp.location.y)
            
        margin = 20
        self.map_bounds = (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)
        
        self.map_surface = pygame.Surface((self.MAP_SIZE, self.MAP_SIZE))
        self.map_surface.fill((20, 20, 30))
        
        for wp in self.waypoints:
            px, py = self._world_to_screen(wp.transform.location.x, wp.transform.location.y)
            pygame.draw.circle(self.map_surface, (60, 60, 80), (px, py), 2)
        
        # Build minimap (250x250) for driving HUD
        self.minimap_surface = pygame.transform.smoothscale(self.map_surface, (250, 250))
        
        print(f"   Mapped {len(self.waypoints)} road points.")

    def _world_to_screen(self, wx, wy):
        """Convert CARLA world coordinates to screen pixels."""
        min_x, max_x, min_y, max_y = self.map_bounds
        px = int((wx - min_x) / (max_x - min_x) * (self.MAP_SIZE - 40) + 20)
        py = int((wy - min_y) / (max_y - min_y) * (self.MAP_SIZE - 40) + 20)
        return px, py

    def _screen_to_world(self, sx, sy):
        """Convert screen pixels back to CARLA world coordinates."""
        min_x, max_x, min_y, max_y = self.map_bounds
        wx = (sx - 20) / (self.MAP_SIZE - 40) * (max_x - min_x) + min_x
        wy = (sy - 20) / (self.MAP_SIZE - 40) * (max_y - min_y) + min_y
        return wx, wy

    def _snap_to_road(self, wx, wy):
        """Find the nearest road waypoint to the clicked location."""
        click_loc = carla.Location(x=wx, y=wy, z=0)
        nearest_wp = self.carla_map.get_waypoint(click_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        return nearest_wp

    def run(self):
        """Main entry: Map selection -> Driving loop."""
        while True:
            destination = self.map_selection_phase()
            if destination is None:
                print("Shutting down...")
                break
            
            continue_running = self.driving_phase(destination)
            if not continue_running:
                break
                
        self._cleanup(full_quit=True)

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1: MAP SELECTION
    # ══════════════════════════════════════════════════════════════
    def map_selection_phase(self):
        """Show top-down map. User clicks a road to set destination."""
        self.display = pygame.display.set_mode((self.MAP_SIZE, self.MAP_SIZE))
        pygame.display.set_caption("Carla Deep Driving")
        
        print("\n MAP MODE: Click on a road to set your destination. Press ENTER to ride!")
        
        # Keep vehicle at current location (persistent between rides)
        if self.vehicle is None or not self.vehicle.is_alive:
            self._spawn_ego()
        
        # Update spawn_wp to vehicle's CURRENT location
        spawn_loc = self.vehicle.get_transform().location
        self.spawn_wp = self.carla_map.get_waypoint(spawn_loc)
        
        selected_dest = None
        clock = pygame.time.Clock()
        
        # UI Button layout
        btn_x = self.MAP_SIZE - 200
        map_btn_y = 50
        weather_btn_y = map_btn_y + 40
        veh_btn_y = weather_btn_y + 40
        ped_btn_y = veh_btn_y + 40
        btn_w = 190
        btn_h = 28

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    
                    # Check Map Dropdown Click
                    if btn_x <= mx <= btn_x + btn_w and map_btn_y <= my <= map_btn_y + btn_h:
                        self.map_dropdown_open = not self.map_dropdown_open
                        continue
                    
                    if self.map_dropdown_open:
                        for i, town in enumerate(self.available_maps):
                            item_y = map_btn_y + btn_h + i * btn_h
                            if btn_x <= mx <= btn_x + btn_w and item_y <= my <= item_y + btn_h:
                                if town != self.current_map_name:
                                    # 1. Destroy ego if alive
                                    if self.vehicle and self.vehicle.is_alive:
                                        self.vehicle.destroy()
                                    
                                    # Clear other actors too
                                    for actor in self.spawned_actors:
                                        try: actor.destroy()
                                        except: pass
                                    self.spawned_actors = []
                                    
                                    # 2. Clear all python references to old CARLA objects to prevent C++ destructor crashes
                                    self.vehicle = None
                                    self.spawn_wp = None
                                    selected_dest = None
                                    self.route = []
                                    self.current_tm_path = []
                                    self.waypoints = []
                                    self.planner = None
                                    self.carla_map = None
                                    self.world = None
                                    
                                    # Force Python garbage collection before the episode dies
                                    import gc
                                    gc.collect()
                                    
                                    # 3. Safely load the new map
                                    self._load_map(town)
                                    self._spawn_ego()
                                    spawn_loc = self.vehicle.get_transform().location
                                    self.spawn_wp = self.carla_map.get_waypoint(spawn_loc)
                                self.map_dropdown_open = False
                                continue
                        self.map_dropdown_open = False
                        continue
                        
                    # Check Weather Click
                    if btn_x <= mx <= btn_x + btn_w and weather_btn_y <= my <= weather_btn_y + btn_h:
                        idx = self.weather_options.index(self.settings['weather'])
                        self.settings['weather'] = self.weather_options[(idx + 1) % len(self.weather_options)]
                        continue
                        
                    # Check Vehicles Click
                    if btn_x <= mx <= btn_x + btn_w and veh_btn_y <= my <= veh_btn_y + btn_h:
                        idx = self.traffic_options.index(self.settings['vehicles'])
                        self.settings['vehicles'] = self.traffic_options[(idx + 1) % len(self.traffic_options)]
                        continue
                        
                    # Check Pedestrians Click
                    if btn_x <= mx <= btn_x + btn_w and ped_btn_y <= my <= ped_btn_y + btn_h:
                        idx = self.pedestrian_options.index(self.settings['pedestrians'])
                        self.settings['pedestrians'] = self.pedestrian_options[(idx + 1) % len(self.pedestrian_options)]
                        continue
                    
                    wx, wy = self._screen_to_world(mx, my)
                    snap = self._snap_to_road(wx, wy)
                    if snap:
                        selected_dest = snap
                        # Compute route preview
                        self.route = self._compute_route(
                            self.vehicle.get_transform().location,
                            snap.transform.location
                        )
                        print(f"   Destination set: ({snap.transform.location.x:.0f}, {snap.transform.location.y:.0f})")

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and selected_dest:
                        print("   APPLYING GOD MODE SETTINGS...")
                        self._apply_god_mode()
                        print("   RIDE STARTED!")
                        return selected_dest

                    if event.key == pygame.K_r:
                        if self.vehicle and self.vehicle.is_alive:
                            self.vehicle.destroy()
                        self._spawn_ego()
                        spawn_loc = self.vehicle.get_transform().location
                        self.spawn_wp = self.carla_map.get_waypoint(spawn_loc)
                        selected_dest = None
                        self.route = []
                        print("   Re-spawned at new location.")

            # ─── Render ───
            self.display.blit(self.map_surface, (0, 0))
            
            # Draw previous breadcrumb trail (dashed blue)
            if len(self.breadcrumb_trail) > 1:
                for i in range(1, len(self.breadcrumb_trail)):
                    if i % 2 == 0:
                        p1 = self._world_to_screen(*self.breadcrumb_trail[i-1])
                        p2 = self._world_to_screen(*self.breadcrumb_trail[i])
                        pygame.draw.line(self.display, (80, 130, 255), p1, p2, 2)
            
            # Draw planned route (red line)
            if len(self.route) > 1:
                for i in range(1, len(self.route)):
                    p1 = self._world_to_screen(*self.route[i-1])
                    p2 = self._world_to_screen(*self.route[i])
                    pygame.draw.line(self.display, (255, 60, 60), p1, p2, 2)
            
            # Draw ego position (blue dot) - pull from live vehicle location
            if self.vehicle and self.vehicle.is_alive:
                ego_loc = self.vehicle.get_transform().location
                ex, ey = self._world_to_screen(ego_loc.x, ego_loc.y)
                pygame.draw.circle(self.display, (0, 150, 255), (ex, ey), 8)
                label = self.font.render("YOU", True, (0, 150, 255))
                self.display.blit(label, (ex + 10, ey - 8))

            # Draw destination (green dot)
            if selected_dest:
                dx, dy = self._world_to_screen(selected_dest.transform.location.x, selected_dest.transform.location.y)
                pygame.draw.circle(self.display, (0, 255, 100), (dx, dy), 8)
                label = self.font.render("DEST", True, (0, 255, 100))
                self.display.blit(label, (dx + 10, dy - 8))

            # God Mode Selector UI
            # 1. Map Dropdown
            btn_color = (80, 80, 120) if not self.map_dropdown_open else (60, 60, 100)
            pygame.draw.rect(self.display, btn_color, (btn_x, map_btn_y, btn_w, btn_h), border_radius=4)
            pygame.draw.rect(self.display, (120, 120, 180), (btn_x, map_btn_y, btn_w, btn_h), 1, border_radius=4)
            map_label = self.font.render(f"{self.current_map_name} v", True, (255, 255, 255))
            self.display.blit(map_label, (btn_x + 8, map_btn_y + 4))
            
            # 2. Weather Toggle
            pygame.draw.rect(self.display, (60, 60, 80), (btn_x, weather_btn_y, btn_w, btn_h), border_radius=4)
            pygame.draw.rect(self.display, (100, 100, 140), (btn_x, weather_btn_y, btn_w, btn_h), 1, border_radius=4)
            w_label = self.font.render(f"Weather: {self.settings['weather']}", True, (200, 230, 255))
            self.display.blit(w_label, (btn_x + 8, weather_btn_y + 4))

            # 3. Vehicles Toggle
            pygame.draw.rect(self.display, (60, 80, 60), (btn_x, veh_btn_y, btn_w, btn_h), border_radius=4)
            pygame.draw.rect(self.display, (100, 140, 100), (btn_x, veh_btn_y, btn_w, btn_h), 1, border_radius=4)
            v_label = self.font.render(f"Traffic: {self.settings['vehicles']} cars", True, (200, 255, 200))
            self.display.blit(v_label, (btn_x + 8, veh_btn_y + 4))

            # 4. Pedestrians Toggle
            pygame.draw.rect(self.display, (80, 60, 60), (btn_x, ped_btn_y, btn_w, btn_h), border_radius=4)
            pygame.draw.rect(self.display, (140, 100, 100), (btn_x, ped_btn_y, btn_w, btn_h), 1, border_radius=4)
            p_label = self.font.render(f"Walkers: {self.settings['pedestrians']}", True, (255, 200, 200))
            self.display.blit(p_label, (btn_x + 8, ped_btn_y + 4))

            if self.map_dropdown_open:
                for i, town in enumerate(self.available_maps):
                    item_y = map_btn_y + btn_h + i * btn_h
                    is_current = (town == self.current_map_name)
                    bg = (40, 100, 60) if is_current else (50, 50, 70)
                    pygame.draw.rect(self.display, bg, (btn_x, item_y, btn_w, btn_h))
                    pygame.draw.rect(self.display, (80, 80, 100), (btn_x, item_y, btn_w, btn_h), 1)
                    txt_color = (0, 255, 100) if is_current else (200, 200, 200)
                    item_surf = self.font.render(f"  {town}" + (" *" if is_current else ""), True, txt_color)
                    self.display.blit(item_surf, (btn_x + 5, item_y + 4))

            # UI Instructions
            title = self.big_font.render("SELECT DESTINATION", True, (255, 255, 255))
            self.display.blit(title, (self.MAP_SIZE // 2 - title.get_width() // 2, 10))
            
            instructions = [
                "LEFT CLICK on a road to set destination",
                "Press ENTER to start ride  |  R = Re-spawn",
            ]
            for i, text in enumerate(instructions):
                surf = self.font.render(text, True, (180, 180, 180))
                self.display.blit(surf, (10, self.MAP_SIZE - 50 + i * 22))

            if selected_dest:
                ready = self.font.render("DESTINATION SET - Press ENTER!", True, (0, 255, 100))
                self.display.blit(ready, (self.MAP_SIZE // 2 - ready.get_width() // 2, 45))

            pygame.display.flip()
            clock.tick(30)

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2: DRIVING TO DESTINATION
    # ══════════════════════════════════════════════════════════════
    def driving_phase(self, destination_wp):
        """Drive to the selected destination using Autopilot + Neural Safety."""
        self.destination_wp = destination_wp
        
        # Resizable driving window
        self.display = pygame.display.set_mode((self.WIN_W, self.WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("Carla Deep Driving - Driving to Destination")

        # Setup sensors
        self._setup_camera()
        self._setup_obstacle_sensor()
        self._setup_collision_sensor()
        
        # Compute planned route for visualization
        self.route = self._compute_route(self.vehicle.get_transform().location, destination_wp.transform.location)
        
        # TM gets confused if points are too dense (loops). Downsample the path (e.g. every 8th point = ~16m apart)
        # This forces the TM to follow OUR smart route logic, but gives it enough breathing room between points.
        self.current_tm_path = [carla.Location(x=p[0], y=p[1], z=0.5) for p in self.route[::8]]
        if self.route: # Ensure the final destination is always the very last point
            self.current_tm_path.append(carla.Location(x=self.route[-1][0], y=self.route[-1][1], z=0.5))
            
        self.breadcrumb_trail = []
        self._breadcrumb_counter = 0
        
        # Wait for camera
        print("Waiting for camera...")
        timeout = time.time() + 10
        while self.current_frame is None and time.time() < timeout:
            self.world.tick()
            time.sleep(0.1)

        # Use CARLA Autopilot with safety tuning
        self.vehicle.set_autopilot(True, self.tm.get_port())
        self.tm.ignore_lights_percentage(self.vehicle, 0.0)
        self.tm.ignore_vehicles_percentage(self.vehicle, 0.0)
        self.tm.distance_to_leading_vehicle(self.vehicle, 5.0)
        self.tm.vehicle_percentage_speed_difference(self.vehicle, 10)
        self.tm.auto_lane_change(self.vehicle, True)
        
        # Set destination via Traffic Manager using the spaced-out smart route
        self.tm.set_path(self.vehicle, self.current_tm_path)
        dest_loc = destination_wp.transform.location
        print(f"Navigating to exact path ({len(self.current_tm_path)} anchors) — route preview has {len(self.route)} pts")
        
        clock = pygame.time.Clock()
        self.stuck_timer = 0
        self.recovering = False
        
        try:
            while True:
                self.world.tick()
                
                if not self.vehicle.is_alive:
                    print("Vehicle destroyed!")
                    return True

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        print("Ride cancelled. Returning to map...")
                        return True
                    if event.type == pygame.VIDEORESIZE:
                        self.WIN_W, self.WIN_H = event.w, event.h
                        self.display = pygame.display.set_mode((self.WIN_W, self.WIN_H), pygame.RESIZABLE)

                v = self.vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                ctrl = self.vehicle.get_control()
                
                # Neural Network inference (shadow mode)
                raw_thr, raw_steer, raw_brake = self._run_inference(speed_kmh)
                
                safety_status = "AUTOPILOT"
                
                # ─── STUCK / CRASH RECOVERY ───
                at_traffic_light = False
                try:
                    at_traffic_light = self.vehicle.is_at_traffic_light()
                except: pass
                
                if self.recovering:
                    elapsed = time.time() - self.recover_start
                    if elapsed < 1.5:
                        ctrl = carla.VehicleControl()
                        ctrl.throttle = 0.4
                        ctrl.steer = 0.0   # Straight reverse
                        ctrl.brake = 0.0
                        ctrl.reverse = True
                        ctrl.hand_brake = False
                        self.vehicle.apply_control(ctrl)
                        safety_status = "REVERSING"
                    else:
                        self.recovering = False
                        self.stuck_timer = 0
                        self.vehicle.set_autopilot(True, self.tm.get_port())
                        self.tm.set_path(self.vehicle, [self.destination_wp.transform.location])
                        safety_status = "RECOVERED"
                        print("   RECOVERED - resuming autopilot")
                else:
                    # Only count stuck if NOT at traffic light
                    if speed_kmh < 0.5 and not at_traffic_light:
                        self.stuck_timer += 1
                    else:
                        self.stuck_timer = max(0, self.stuck_timer - 3)
                    
                    # Reverse only after 15s stuck AND not at traffic light
                    if self.stuck_timer > 300:
                        print("   Car stuck for 15s! Auto-reversing...")
                        self.recovering = True
                        self.recover_start = time.time()
                        self.vehicle.set_autopilot(False)
                        self.stuck_timer = 0
                    
                    # Radar safety (ignore < 1.5m = self-detection)
                    if 1.5 < self.obstacle_dist < 3.0:
                        ctrl.throttle = 0.0
                        ctrl.brake = 1.0
                        safety_status = "EMERGENCY STOP"
                        self.vehicle.apply_control(ctrl)
                    elif 3.0 <= self.obstacle_dist < 6.0:
                        ctrl.throttle = min(ctrl.throttle, 0.3)
                        ctrl.brake = max(ctrl.brake, 0.2)
                        safety_status = "CAUTION"
                        self.vehicle.apply_control(ctrl)
                    
                    if at_traffic_light:
                        safety_status = "RED LIGHT"
                
                # Check if arrived
                ego_loc = self.vehicle.get_transform().location
                dist_to_dest = ego_loc.distance(self.destination_wp.transform.location)
                
                if dist_to_dest < 5.0:
                    ctrl.throttle = 0.0
                    ctrl.brake = 1.0
                    self.vehicle.apply_control(ctrl)
                    print(f"\nARRIVED AT DESTINATION! Distance: {dist_to_dest:.1f}m")
                    
                    # Show arrived screen
                    overlay = pygame.Surface((self.WIN_W, self.WIN_H))
                    overlay.set_alpha(150)
                    overlay.fill((0, 0, 0))
                    self.display.blit(overlay, (0, 0))
                    
                    done_text = self.big_font.render("ARRIVED! Returning to map...", True, (0, 255, 100))
                    self.display.blit(done_text, (self.WIN_W // 2 - done_text.get_width() // 2, self.WIN_H // 2))
                    pygame.display.flip()
                    
                    # Ensure Pygame doesn't freeze or lag while sleeping
                    for _ in range(30):
                        pygame.event.pump()
                        time.sleep(0.1)
                        
                    return True

                # Decay radar
                self.obstacle_dist = min(50.0, self.obstacle_dist + 0.3)

                # Record breadcrumb trail
                self._breadcrumb_counter += 1
                if self._breadcrumb_counter % 5 == 0:
                    self.breadcrumb_trail.append((ego_loc.x, ego_loc.y))

                # Render
                self._render_driving(speed_kmh, ctrl, safety_status, dist_to_dest, raw_thr, raw_steer, raw_brake)
                
                # Telemetry logging
                self.log_writer.writerow([
                    f"{time.time():.3f}", f"{speed_kmh:.2f}", 
                    f"{ctrl.throttle:.2f}", f"{ctrl.brake:.2f}",
                    f"{ctrl.steer:.2f}", f"{self.obstacle_dist:.1f}", safety_status
                ])

                if int(time.time() * 5) % 3 == 0:
                    print(f"Speed: {speed_kmh:.1f} km/h | Dist: {dist_to_dest:.0f}m | Gap: {self.obstacle_dist:.1f}m | {safety_status}")

                clock.tick(20)
        finally:
            self._cleanup(full_quit=False)

    # ══════════════════════════════════════════════════════════════
    #  MAP / WORLD HELPERS
    # ══════════════════════════════════════════════════════════════
    def _load_map(self, town_name):
        """Load a new CARLA map and rebuild the road network."""
        print(f"\nLoading {town_name}... (this may take a moment)")

        
        self.display.fill((20, 20, 30))
        loading_text = self.big_font.render(f"Loading {town_name}...", True, (255, 255, 255))
        self.display.blit(loading_text, (self.MAP_SIZE // 2 - loading_text.get_width() // 2, self.MAP_SIZE // 2))
        pygame.display.flip()
        
        self.client.load_world(town_name)
        time.sleep(8)
        
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(False)
        
        # Re-initialize planner for the new map
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            self.planner = GlobalRoutePlanner(self.carla_map, 2.0)
        except:
            self.planner = None
            
        self.current_map_name = town_name
        self._build_map_data()
        print(f"   {town_name} loaded successfully!")

    def _compute_route(self, start_loc, end_loc):
        """Trace road waypoints using GlobalRoutePlanner (A*)."""
        route_points = []
        
        # Use smart planner if available
        if self.planner:
            try:
                # Snap start/end to nearest road waypoints to ensure a valid route
                s_wp = self.carla_map.get_waypoint(start_loc, project_to_road=True)
                e_wp = self.carla_map.get_waypoint(end_loc, project_to_road=True)
                
                if s_wp and e_wp:
                    # Returns list of (waypoint, RoadOption)
                    path = self.planner.trace_route(s_wp.transform.location, e_wp.transform.location)
                    for wp, _ in path:
                        route_points.append((wp.transform.location.x, wp.transform.location.y))
                    return route_points
            except Exception as e:
                print(f"   Smart path error: {e}. Falling back to greedy...")

        # Fallback to greedy search (if planner fails or is missing)
        try:
            wp = self.carla_map.get_waypoint(start_loc, project_to_road=True)
            end_wp = self.carla_map.get_waypoint(end_loc, project_to_road=True)
            if not wp or not end_wp:
                return route_points
            
            route_points.append((wp.transform.location.x, wp.transform.location.y))
            
            for _ in range(1000):
                next_wps = wp.next(2.0)
                if not next_wps:
                    break
                
                best_wp = None
                best_dist = float('inf')
                for nwp in next_wps:
                    d = nwp.transform.location.distance(end_loc)
                    if d < best_dist:
                        best_dist = d
                        best_wp = nwp
                
                if best_wp is None:
                    break
                    
                wp = best_wp
                route_points.append((wp.transform.location.x, wp.transform.location.y))
                
                if best_dist < 5.0:
                    route_points.append((end_loc.x, end_loc.y))
                    break
        except Exception as e:
            print(f"   Greedy route error: {e}")
        
        return route_points

    def _spawn_ego(self):
        """Spawn ego vehicle at a random location."""
        bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        spawn_points = self.carla_map.get_spawn_points()
        np.random.shuffle(spawn_points)
        for sp in spawn_points:
            self.vehicle = self.world.try_spawn_actor(bp, sp)
            if self.vehicle:
                print(f"   Ego spawned at ({sp.location.x:.0f}, {sp.location.y:.0f})")
                return
        raise RuntimeError("Could not spawn ego vehicle!")

    def _apply_god_mode(self):
        """Applies dynamic weather, spawns requested background traffic and pedestrians."""
        print(f"   [GOD MODE] Weather: {self.settings['weather']} | Cars: {self.settings['vehicles']} | Walkers: {self.settings['pedestrians']}")
        
        # 1. Set Weather
        w = self.settings['weather']
        if w == 'Clear':
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        elif w == 'Rain':
            self.world.set_weather(carla.WeatherParameters.HeavyRainNoon)
        elif w == 'Night':
            self.world.set_weather(carla.WeatherParameters.ClearNight)
        elif w == 'Fog':
            wp = carla.WeatherParameters.ClearNoon
            wp.fog_density = 90.0
            wp.fog_distance = 0.0
            self.world.set_weather(wp)
            
        # 2. Spawn Vehicles
        num_veh = self.settings['vehicles']
        if num_veh > 0:
            spawn_points = self.carla_map.get_spawn_points()
            np.random.shuffle(spawn_points)
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')
            blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]
            spawned = 0
            for sp in spawn_points:
                if spawned >= num_veh: break
                bp = np.random.choice(blueprints)
                if bp.has_attribute('color'):
                    color = np.random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)
                bp.set_attribute('role_name', 'autopilot')
                veh = self.world.try_spawn_actor(bp, sp)
                if veh is not None:
                    veh.set_autopilot(True, self.tm.get_port())
                    self.spawned_actors.append(veh)
                    spawned += 1

        # 3. Spawn Pedestrians
        num_ped = self.settings['pedestrians']
        if num_ped > 0:
            blueprintsWalkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            for i in range(num_ped):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    sp = carla.Transform(loc)
                    bp = np.random.choice(blueprintsWalkers)
                    if bp.has_attribute('is_invincible'):
                        bp.set_attribute('is_invincible', 'false')
                    walker = self.world.try_spawn_actor(bp, sp)
                    if walker:
                        self.spawned_actors.append(walker)
                        controller = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
                        if controller:
                            self.spawned_actors.append(controller)
                            controller.start()
                            controller.go_to_location(self.world.get_random_location_from_navigation())
                            controller.set_max_speed(1.4 + np.random.random())

    # ══════════════════════════════════════════════════════════════
    #  SENSORS
    # ══════════════════════════════════════════════════════════════
    def _setup_camera(self):
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.CAM_W))
        cam_bp.set_attribute('image_size_y', str(self.CAM_H))
        cam_bp.set_attribute('fov', '100')
        cam_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: self._on_image(data))

    def _setup_obstacle_sensor(self):
        obs_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
        obs_bp.set_attribute('distance', '15')
        obs_bp.set_attribute('hit_radius', '0.5')
        obs_transform = carla.Transform(carla.Location(x=2.5, z=0.8))
        self.obstacle_sensor = self.world.spawn_actor(obs_bp, obs_transform, attach_to=self.vehicle)
        self.obstacle_sensor.listen(lambda data: self._on_obstacle(data))

    def _setup_collision_sensor(self):
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.current_frame = array[:, :, :3]

    def _on_collision(self, event):
        self.last_collision_time = time.time()
        try:
            v = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            if speed < 2.0 and not self.recovering:
                print("   COLLISION while stopped! Auto-recovering...")
                self.recovering = True
                self.recover_start = time.time()
                self.vehicle.set_autopilot(False)
        except: pass

    def _on_obstacle(self, data):
        # Filter out self-detections (< 1.5m is car body / ground)
        if data.distance < 1.5:
            return
        if data.distance < self.obstacle_dist:
            self.obstacle_dist = data.distance
        else:
            self.obstacle_dist = self.obstacle_dist * 0.9 + data.distance * 0.1

    # ══════════════════════════════════════════════════════════════
    #  NEURAL NETWORK
    # ══════════════════════════════════════════════════════════════
    def _run_inference(self, speed_kmh):
        """Run neural network inference. Returns (throttle, steer, brake)."""
        if self.current_frame is None:
            return 0.0, 0.0, 0.0
        try:
            img = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            speed_t = torch.tensor([[speed_kmh / 100.0]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(img_t, speed_t)[0]
            return float(output[0]), float(output[1]), float(output[2])
        except:
            return 0.0, 0.0, 0.0

    # ══════════════════════════════════════════════════════════════
    #  RENDERING
    # ══════════════════════════════════════════════════════════════
    def _render_driving(self, speed, ctrl, status, dist_to_dest, ai_thr, ai_steer, ai_brake):
        """Render camera view + HUD + minimap."""
        if self.current_frame is None:
            return
        
        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(cam_surface, (self.WIN_W, self.WIN_H))
        self.display.blit(scaled, (0, 0))

        # ─── LEFT HUD PANEL ───
        overlay = pygame.Surface((360, 280))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.display.blit(overlay, (10, 10))

        if "EMERGENCY" in status or "REVERSING" in status:
            status_color = (255, 50, 50)
        elif "CAUTION" in status:
            status_color = (255, 200, 0)
        elif "RECOVERED" in status:
            status_color = (0, 200, 255)
        elif "RED LIGHT" in status:
            status_color = (255, 100, 100)
        else:
            status_color = (0, 255, 100)

        lines = [
            ("CARLA DEEP DRIVING", (100, 200, 255)),
            (f"Status: {status}", status_color),
            (f"Speed: {speed:.1f} km/h", (255, 255, 255)),
            (f"Distance to Dest: {dist_to_dest:.0f}m", (255, 255, 0)),
            (f"Radar Gap: {self.obstacle_dist:.1f}m", (255, 255, 255)),
            ("--- Autopilot ---", (100, 100, 100)),
            (f"Throttle: {ctrl.throttle:.2f}  Brake: {ctrl.brake:.2f}", (255, 255, 255)),
            (f"Steer: {ctrl.steer:.2f}", (255, 255, 255)),
            ("--- AI Brain ---", (100, 100, 100)),
            (f"AI T:{ai_thr:.2f}  B:{ai_brake:.2f}  S:{ai_steer:.2f}", (180, 180, 255)),
            ("[ESC] Cancel  [X] Quit", (120, 120, 120)),
        ]

        for i, (text, color) in enumerate(lines):
            surf = self.font.render(text, True, color)
            self.display.blit(surf, (20, 20 + i * 24))

        # ─── MINIMAP (top-right corner) ───
        minimap_size = 250
        mm_x = self.WIN_W - minimap_size - 15
        mm_y = 15
        scale = minimap_size / self.MAP_SIZE
        
        pygame.draw.rect(self.display, (40, 40, 50), (mm_x - 3, mm_y - 3, minimap_size + 6, minimap_size + 6), border_radius=5)
        
        mm = self.minimap_surface.copy()
        
        # Draw planned route (red)
        if len(self.route) > 1:
            for i in range(1, len(self.route)):
                p1 = self._world_to_screen(*self.route[i-1])
                p2 = self._world_to_screen(*self.route[i])
                mp1 = (int(p1[0] * scale), int(p1[1] * scale))
                mp2 = (int(p2[0] * scale), int(p2[1] * scale))
                pygame.draw.line(mm, (255, 60, 60), mp1, mp2, 2)
        
        # Draw breadcrumb trail (dashed blue)
        if len(self.breadcrumb_trail) > 1:
            for i in range(1, len(self.breadcrumb_trail)):
                if i % 2 == 0:
                    p1 = self._world_to_screen(*self.breadcrumb_trail[i-1])
                    p2 = self._world_to_screen(*self.breadcrumb_trail[i])
                    mp1 = (int(p1[0] * scale), int(p1[1] * scale))
                    mp2 = (int(p2[0] * scale), int(p2[1] * scale))
                    pygame.draw.line(mm, (80, 130, 255), mp1, mp2, 2)
        
        # Ego position (blue dot)
        if self.vehicle and self.vehicle.is_alive:
            ego_loc = self.vehicle.get_transform().location
            ex, ey = self._world_to_screen(ego_loc.x, ego_loc.y)
            pygame.draw.circle(mm, (0, 150, 255), (int(ex * scale), int(ey * scale)), 5)
        
        # Destination (green dot)
        if self.destination_wp:
            dl = self.destination_wp.transform.location
            dx, dy = self._world_to_screen(dl.x, dl.y)
            pygame.draw.circle(mm, (0, 255, 100), (int(dx * scale), int(dy * scale)), 5)
        
        self.display.blit(mm, (mm_x, mm_y))
        
        mm_label = self.font.render(f"MAP - {self.current_map_name}", True, (200, 200, 200))
        self.display.blit(mm_label, (mm_x, mm_y + minimap_size + 5))
        
        pygame.display.flip()

    # ══════════════════════════════════════════════════════════════
    #  CLEANUP
    # ══════════════════════════════════════════════════════════════
    def _cleanup(self, full_quit=False):
        print("Cleaning up sensors...")
        try:
            if self.camera and self.camera.is_alive: self.camera.destroy()
            if self.obstacle_sensor and self.obstacle_sensor.is_alive: self.obstacle_sensor.destroy()
            if self.collision_sensor and self.collision_sensor.is_alive: self.collision_sensor.destroy()
            
            self.camera = None
            self.obstacle_sensor = None
            self.collision_sensor = None
            self.current_frame = None
            
            # Clean up God Mode actors (background traffic/pedestrians)
            for actor in self.spawned_actors:
                try: 
                    # If it's a walker AI controller, stop it before destroying
                    if hasattr(actor, 'stop'): actor.stop()
                    actor.destroy()
                except: pass
            self.spawned_actors.clear()
            
            # Only destroy vehicle on full quit (persistent between rides)
            if full_quit:
                if self.vehicle and self.vehicle.is_alive: self.vehicle.destroy()
                self.vehicle = None
        except: pass
        
        if full_quit:
            print("Exiting Carla Deep Driving.")
            if hasattr(self, 'log_file'): self.log_file.close()
            pygame.quit()


if __name__ == "__main__":
    pilot = GTAPilot()
    pilot.run()
