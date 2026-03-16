import carla
import random
import time
import math

class IndianTrafficManager:
    def __init__(self, client):
        self.client = client
        self.world = client.get_world()
        self.tm = client.get_trafficmanager(8000)
        self.spawned_vehicles = []
        self.spawned_walkers = []
        self.walker_controllers = []

        # Configure Traffic Manager for "Indian Style"
        self.tm.set_synchronous_mode(True)
        # 1.5m - 2.0m is "Dense" but "Safe" for CARLA physics
        self.tm.set_global_distance_to_leading_vehicle(2.0)
        # Enable hybrid mode for performance with many actors
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_hybrid_physics_radius(70.0)

    def spawn_traffic(self, num_vehicles=100):
        blueprint_library = self.world.get_blueprint_library()
        blueprints = blueprint_library.filter('vehicle.*')
        
        # Exclude huge trucks/trailers for more fluid "Indian" traffic
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) >= 2 and 'firetruck' not in x.id and 'ambulance' not in x.id]
        
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        count = 0
        for n, transform in enumerate(spawn_points):
            if count >= num_vehicles:
                break
            
            bp = random.choice(blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            
            vehicle = self.world.try_spawn_actor(bp, transform)
            if vehicle:
                vehicle.set_autopilot(True, self.tm.get_port())
                
                # --- UNIQUE INDIAN BEHAVIOR SETTINGS ---
                
                # 1. Lane Indiscipline (Lane Offset)
                # This makes vehicles drive slightly off-center, simulating "squeezing" and lane sharing
                offset = random.uniform(-0.8, 0.8) 
                self.tm.vehicle_lane_offset(vehicle, offset)
                
                # 2. Speed Variance
                # Some drivers are slow (rickshaw-style), some are fast
                speed_diff = random.uniform(-30, 10) 
                self.tm.vehicle_percentage_speed_difference(vehicle, speed_diff)
                
                # 3. Aggressive Overtaking
                self.tm.auto_lane_change(vehicle, True)
                self.tm.random_left_lanechange_percentage(vehicle, 40)
                self.tm.random_right_lanechange_percentage(vehicle, 40)
                
                # 4. Tailgating (but safe)
                self.tm.distance_to_leading_vehicle(vehicle, random.uniform(1.5, 3.0))
                
                # 5. Occasional Traffic Rule "Fluidity" (very low percentage to avoid "dumb" crashes)
                self.tm.ignore_lights_percentage(vehicle, 5) # 5% chance to jump yellow/red
                self.tm.ignore_signs_percentage(vehicle, 5)

                self.spawned_vehicles.append(vehicle)
                count += 1

        print(f"Spawned {len(self.spawned_vehicles)} vehicles with Indian behavior.")

    def spawn_pedestrians(self, num_walkers=50):
        blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        for i in range(num_walkers):
            # 1. Get random location on sidewalks
            loc = self.world.get_random_location_from_navigation()
            if loc:
                sp = carla.Transform(loc)
                bp = random.choice(blueprints)
                
                walker = self.world.try_spawn_actor(bp, sp)
                if walker:
                    self.spawned_walkers.append(walker)
                    
                    controller = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    if controller:
                        self.walker_controllers.append(controller)
                        controller.start()
                        controller.go_to_location(self.world.get_random_location_from_navigation())
                        controller.set_max_speed(1.0 + random.random()) # Human walking speed

        print(f"Spawned {len(self.spawned_walkers)} pedestrians.")

    def cleanup(self):
        print("Cleaning up Indian Traffic...")
        for controller in self.walker_controllers:
            controller.stop()
        
        all_actors = self.spawned_vehicles + self.spawned_walkers + self.walker_controllers
        for actor in all_actors:
            if actor.is_alive:
                actor.destroy()
        print("Cleanup complete.")

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Ensure synchronous mode for clean physics
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        manager = IndianTrafficManager(client)
        manager.spawn_traffic(120) # Dense but manageable
        manager.spawn_pedestrians(60)

        print("🚦 Indian Traffic Simulator Active (Lane Sharing + Varied Speeds) 🚦")
        print("Press Ctrl+C to stop.")

        while True:
            world.tick()
            
    except KeyboardInterrupt:
        pass
    finally:
        manager.cleanup()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    main()
