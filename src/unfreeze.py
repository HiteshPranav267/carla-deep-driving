import carla
import time

def force_async():
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(5.0)
        print("Force unFreezing simulator...")
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print("Simulator Unfrozen successfully.")
    except Exception as e:
        print(f"Failed to unfreeze: {e}")

if __name__ == "__main__":
    force_async()
