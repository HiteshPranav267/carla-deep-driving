import carla
import random
import time
import math
import logging

def spawn_pedestrians(client, world, num_walkers):
    blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
    spawn_points = []
    
    for _ in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)

    batch = []
    walker_speed = []
    
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # make invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        
        # speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > 0.5):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1]) # walk
            else:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2]) # run
        else:
            walker_speed.append(0.0)
            
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
    results = client.apply_batch_sync(batch, True)
    
    walker_ids = []
    speeds = []
    for i, res in enumerate(results):
        if not res.error:
            walker_ids.append(res.actor_id)
            speeds.append(walker_speed[i])

    # AI controllers
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for walker_id in walker_ids:
        batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker_id))
        
    results = client.apply_batch_sync(batch, True)
    controller_ids = []
    for res in results:
        if not res.error:
            controller_ids.append(res.actor_id)

    world.tick()

    all_actors = world.get_actors(controller_ids)
    
    # 30% of pedestrians will blindly cross roads
    world.set_pedestrians_cross_factor(30.0)
    
    for i, controller in enumerate(all_actors):
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(float(speeds[i]))
        
    print(f"Spawned {len(walker_ids)} pedestrians.")
    return walker_ids, controller_ids


def spawn_traffic(client, world, tm, num_vehicles):
    vehicle_bps = world.get_blueprint_library().filter("vehicle.*")
    safe_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) >= 2]
    
    bikes = [bp for bp in safe_bps if int(bp.get_attribute('number_of_wheels')) == 2]
    cars = [bp for bp in safe_bps if int(bp.get_attribute('number_of_wheels')) > 2]
    
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    vehicles = []
    bike_actors = []
    
    batch = []
    for i, sp in enumerate(spawn_points[:num_vehicles]):
        # 40% bikes, 60% cars
        bp = random.choice(bikes) if random.random() < 0.4 else random.choice(cars)
        
        batch.append(carla.command.SpawnActor(bp, sp).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())
        ))

    results = client.apply_batch_sync(batch, True)
    
    for res in results:
        if not res.error:
            v = world.get_actor(res.actor_id)
            vehicles.append(v)
            if v.attributes.get('number_of_wheels') == '2':
                bike_actors.append(v)
            
            # --- INDIAN TRAFFIC BEHAVIOR PARAMETERS ---
            # 1. Very close distance to leading vehicle (gap exploitation)
            tm.distance_to_leading_vehicle(v, random.uniform(0.1, 1.0))
            # 2. Low random lane changing to keep them on road waypoints
            tm.random_left_lanechange_percentage(v, 10)
            tm.random_right_lanechange_percentage(v, 10)
            # 3. Respect signals as requested
            tm.ignore_lights_percentage(v, 0)
            tm.ignore_signs_percentage(v, 0)
            # 4. Global Safety - Hard Brake if needed
            tm.ignore_vehicles_percentage(v, 2.0) # Only 2% chance to nudge
            tm.distance_to_leading_vehicle(v, 3.0) 
            # 5. Speed variance
            tm.vehicle_percentage_speed_difference(v, random.uniform(-15, 5))


    print(f"Spawned {len(vehicles)} vehicles (Bikes: {len(bike_actors)})")
    return vehicles, bike_actors

def spawn_static_obstacles(world, num_clusters):
    props = world.get_blueprint_library().filter("static.prop.*")
    debris_props = [p for p in props if "trash" in p.id or "box" in p.id or "barrel" in p.id or "barrier" in p.id]
    if not debris_props:
        return []
    
    waypoints = world.get_map().generate_waypoints(20.0)
    random.shuffle(waypoints)
    
    obstacles = []
    # Only use a fraction of waypoints to create distinct "piles"
    for wp in waypoints[:num_clusters]:
        # 1. Determine road side (shift 6.5m to be safely on the sidewalk/verge)
        right_vec = wp.transform.get_right_vector()
        base_loc = wp.transform.location + carla.Location(x=right_vec.x * 6.5, y=right_vec.y * 6.5, z=0.2)
        
        # 2. Spawn 3-5 items in a cluster
        for _ in range(random.randint(3, 5)):
            # Random offset within the pile
            offset = carla.Location(x=random.uniform(-0.8, 0.8), y=random.uniform(-0.8, 0.8), z=0.1)
            spawn_loc = base_loc + offset
            spawn_rot = carla.Rotation(yaw=random.uniform(0, 360))
            
            bp = random.choice(debris_props)
            obs = world.try_spawn_actor(bp, carla.Transform(spawn_loc, spawn_rot))
            if obs:
                obstacles.append(obs)
            
    print(f"Spawned {len(obstacles)} items in {num_clusters} roadside piles.")
    return obstacles


def manage_chaos(world, tm, vehicles, bikes, stuck_tracker):
    """
    Called every few ticks to enforce Indian traffic heuristics:
    1. Gridlock resolution (Clean cleanup, no more reversing)
    2. Center-lane riding for bikes
    3. Sudden bizarre braking (Occasional)
    """
    map = world.get_map()
    now = time.time()
    
    for v in vehicles:
        if not v.is_alive:
            continue
            
        # --- FIX: Periodically reset speed diff to prevent permanent "Brake" stuckness ---
        if random.random() < 0.02:
            tm.vehicle_percentage_speed_difference(v, random.uniform(-20, 10))

        vel = v.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2)
        
        # Simple Anti-Gridlock: Just destroy if blocked too long
        if speed < 0.1:
            if v.id not in stuck_tracker:
                stuck_tracker[v.id] = now
            elif now - stuck_tracker[v.id] > 30.0:  # 30s threshold
                print(f"Vehicle {v.id} timed out. Destroying to clear road.")
                v.destroy()
                stuck_tracker.pop(v.id)
        else:
            if v.id in stuck_tracker:
                stuck_tracker.pop(v.id)
                
    # 2. Bike center line riding (weaving)
    for bike in bikes:
        if not bike.is_alive:
            continue
        # 10% chance to force it to weave to the lane marker
        if random.random() < 0.05:
            wp = map.get_waypoint(bike.get_location())
            if wp:
                # shift to right edge of lane to simulate squeezing between lanes
                right_v = wp.transform.get_right_vector()
                loc = wp.transform.location + carla.Location(x=right_v.x*1.5, y=right_v.y*1.5)
                # Apply a slight steering offset via control (more natural than set_transform)
                # Since tm overrides steering, we tell TM to change lane right immediately
                tm.force_lane_change(bike, True)

    # 3. Wrong side driving
    for v in vehicles:
        if not v.is_alive:
            continue
        vel = v.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2)
        
        # If speed is low, 2% chance to switch to opposite lane to overtake
        if speed < 1.0 and random.random() < 0.02:
            wp = map.get_waypoint(v.get_location())
            if wp and wp.get_left_lane():
                tm.force_lane_change(v, False) # force left lane change
                
    # 4. Sudden braking
    for v in vehicles:
        if not v.is_alive: continue
        if random.random() < 0.005: 
            # Force severe brake in Traffic Manager
            tm.vehicle_percentage_speed_difference(v, 100) # Stop
            stuck_tracker[v.id] = now  # Use simple timestamp
            

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        
        # Load town with high complexity like Town03 or Town05
        # world = client.load_world("Town03")
        
        tm = client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(0.5)
        tm.set_hybrid_physics_mode(False)
        tm.set_synchronous_mode(True)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Entities
        vehicles, bikes = spawn_traffic(client, world, tm, 100)
        walkers, walker_controllers = spawn_pedestrians(client, world, 50)
        
        stuck_tracker = {}

        print("🚦 Indian Traffic Chaos Simulation Running... 🚦")
        
        while True:
            world.tick()
            manage_chaos(world, tm, vehicles, bikes, stuck_tracker)
            
            # Reset random sudden brakers to normal speed randomly
            for v in vehicles:
                if v.is_alive and random.random() < 0.01:
                    tm.vehicle_percentage_speed_difference(v, random.uniform(-15, 5))

    except KeyboardInterrupt:
        print("Cleaning up...")
    finally:
        # Restore settings
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers])
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_controllers])
        time.sleep(0.5)

if __name__ == '__main__':
    main()
