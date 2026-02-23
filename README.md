# End-to-End Deep Learning Autonomous Driving for Indian Traffic (CARLA)

## Core Research Problem
Developing a fully deep learning-based predictive driving model that operates in complex, semi-structured, rule-flexible traffic environments (simulating Indian traffic) without relying on explicit rule-based components or hand-coded symbolic logic.

## Project Structure
`src/`
- `indian_traffic_manager.py` - Custom traffic generator. Simulates high density, lack of lane discipline, bike-centric behavior, wrong-side driving, pedestrian jaywalking, debris, and anti-gridlock mechanisms.
- `data_collector.py` - Script to log RGB, LiDAR, Ego state, and human/autopilot control actions to create a training dataset.
- `model.py` - Deep Neural Network architecture (CNN/Transformer for vision + MLP for state -> continuous control actions: throttle, steer, brake).
- `train.py` - PyTorch training script mapping observations to expert actions (Imitation Learning/Behavioral Cloning).
- `autonomous_agent.py` - Evaluation script deploying the trained model in CARLA without any hand-crafted rules or waypoints.

### Why The Previous Traffic Failed & How We Fixed It:
1. **Gridlocks (Cars stuck randomly):** We added an `Anti-Gridlock Tracker` that monitors car velocities. If a car is stuck (speed ~ 0) for over 15 seconds, it is teleported or destroyed/respawned to maintain flow.
2. **Signals Stuck:** We force traffic to ignore traffic lights and stop signs at a high percentage (matching real chaotic junctions), or actively turn traffic lights green/off. We use `tm.ignore_lights_percentage(actor, 90)`.
3. **Boring/Line behavior:** We force lane changes (`random_left_lanechange_percentage`), gap exploitation (low minimum distance), and specifically spawn bikes that actively ignore lane centers to weave through traffic.
4. **Debris:** We spawn static props randomly on the road edges.

## Guide to Running the Project

### Phase 1: Data Collection (Imitation Learning)
The model needs to learn from an "expert" navigating this chaos. We will use CARLA's built-in autopilot (configured securely, or heavily tweaked) OR human driving mode to collect data.
1. Start CARLA server.
2. Run `python src/indian_traffic_manager.py` (Leave it running in the background to keep the environment chaotic).
3. Run `python src/data_collector.py` to spawn the Ego vehicle and record images and controls to `dataset/`.

### Phase 2: Training the Neural Network
We train the model to map *Raw Inputs (Camera + Speed)* directly to *Control Actions (Steering, Throttle, Braking)*.
1. Run `python src/train.py`.
2. This will save a `best_model.pth`.

### Phase 3: Rule-Free Evaluation
We evaluate the model in the simulator. The vehicle relies purely on the DL model predictions—no A*, no MPC, no behavior trees, no PID controllers for trajectory tracking.
1. Start CARLA server.
2. Run `python src/indian_traffic_manager.py`.
3. Run `python src/autonomous_agent.py` to deploy the purely neural agent. Observe its emergent behavior.

---
Let's build!
