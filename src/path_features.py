"""
path_features.py — Shared path-feature computation for data collection and inference.

Computes route-relative features each tick:
  - Heading deltas to next 5 route waypoints
  - Curvature at near / mid / far horizon
  - Distance to next junction
  - Turn intent (left / straight / right)
  - Traffic light class (red / yellow / green / none)
  - Stop-required flag
"""

import math
import numpy as np

try:
    import carla
except ImportError:
    carla = None  # allow offline import for dataset tools


# -----------------------------------------------------------------------
#  Heading delta
# -----------------------------------------------------------------------
def heading_delta(ego_fwd, wp):
    """Signed angle from ego forward to waypoint forward (radians)."""
    wp_fwd = wp.transform.get_forward_vector()
    cross = ego_fwd.x * wp_fwd.y - ego_fwd.y * wp_fwd.x
    dot = ego_fwd.x * wp_fwd.x + ego_fwd.y * wp_fwd.y
    return math.atan2(cross, dot)


def heading_deltas(ego_fwd, route_wps, n=5):
    """
    Compute heading delta to next *n* route waypoints.
    Returns list of n floats (pad with 0.0 if fewer waypoints remain).
    """
    deltas = []
    for i in range(n):
        if i < len(route_wps):
            wp = route_wps[i]
            # route_wps items may be (waypoint, command) tuples
            if isinstance(wp, tuple):
                wp = wp[0]
            deltas.append(heading_delta(ego_fwd, wp))
        else:
            deltas.append(0.0)
    return deltas


# -----------------------------------------------------------------------
#  Curvature estimation
# -----------------------------------------------------------------------
def _curvature_from_3pts(p1, p2, p3):
    """Menger curvature from 3 CARLA locations. Returns 1/R."""
    ax, ay = p1.x, p1.y
    bx, by = p2.x, p2.y
    cx, cy = p3.x, p3.y

    # area of triangle * 2
    area2 = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    ab = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2) + 1e-8
    bc = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2) + 1e-8
    ac = math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) + 1e-8
    return (2.0 * area2) / (ab * bc * ac + 1e-8)


def curvature_at_horizon(anchor_wp, dist_start, dist_end, n_samples=3):
    """
    Average curvature between *dist_start* m and *dist_end* m ahead of anchor_wp.
    Samples n_samples waypoints.
    """
    step = max(1.0, (dist_end - dist_start) / max(1, n_samples - 1))
    pts = []
    wp = anchor_wp
    # CARLA requires distance > 0.0 for waypoint.next().
    start = max(float(dist_start), 1e-3)
    for d in np.arange(start, dist_end + 0.1, step):
        nexts = wp.next(step) if pts else anchor_wp.next(float(max(d, 1e-3)))
        if nexts:
            wp = nexts[0]
            pts.append(wp.transform.location)
        if len(pts) >= n_samples + 1:
            break

    if len(pts) < 3:
        return 0.0

    curvs = []
    for i in range(len(pts) - 2):
        curvs.append(_curvature_from_3pts(pts[i], pts[i + 1], pts[i + 2]))
    return float(np.mean(curvs)) if curvs else 0.0


def curvature_features(anchor_wp):
    """Returns (curvature_near, curvature_mid, curvature_far)."""
    near = curvature_at_horizon(anchor_wp, 0, 10, n_samples=4)
    mid = curvature_at_horizon(anchor_wp, 10, 30, n_samples=4)
    far = curvature_at_horizon(anchor_wp, 30, 60, n_samples=4)
    return near, mid, far


# -----------------------------------------------------------------------
#  Junction distance
# -----------------------------------------------------------------------
def dist_to_next_junction(anchor_wp, max_dist=100.0, step=3.0):
    """Walk forward from anchor_wp and return distance to first junction."""
    total = 0.0
    wp = anchor_wp
    while total < max_dist:
        nexts = wp.next(step)
        if not nexts:
            break
        wp = nexts[0]
        total += step
        if wp.is_junction:
            return float(total)
    return float(max_dist)


# -----------------------------------------------------------------------
#  Turn intent from route command
# -----------------------------------------------------------------------
# CARLA RoadOption enum values
_LEFT_CMDS = {'Left', 'LaneChangeLeft'}
_RIGHT_CMDS = {'Right', 'LaneChangeRight'}
_STRAIGHT_CMDS = {'Straight', 'RoadOption.STRAIGHT'}

# Numeric RoadOption values (if using int encoding)
# 1=Left, 2=Right, 3=Straight, 4=FollowLane, 5=ChangeLaneLeft, 6=ChangeLaneRight
_LEFT_INTS = {1, 5}
_RIGHT_INTS = {2, 6}
_STRAIGHT_INTS = {3, 4}


def turn_intent_from_command(command):
    """
    Convert CARLA route command to turn intent string.
    Accepts RoadOption enum, its string name, or int value.
    Returns: 'left', 'straight', or 'right'.
    """
    if command is None:
        return 'straight'

    # Handle CARLA RoadOption enum
    if hasattr(command, 'name'):
        name = command.name
    elif isinstance(command, int):
        if command in _LEFT_INTS:
            return 'left'
        elif command in _RIGHT_INTS:
            return 'right'
        else:
            return 'straight'
    else:
        name = str(command)

    if name in _LEFT_CMDS or 'Left' in name:
        return 'left'
    elif name in _RIGHT_CMDS or 'Right' in name:
        return 'right'
    else:
        return 'straight'


def turn_intent_from_route(route_wps_with_cmds, current_idx, lookahead=5):
    """
    Look ahead in the route for the next non-straight command.
    Returns the first upcoming turn intent.
    """
    for i in range(current_idx, min(current_idx + lookahead, len(route_wps_with_cmds))):
        _, cmd = route_wps_with_cmds[i]
        intent = turn_intent_from_command(cmd)
        if intent != 'straight':
            return intent
    return 'straight'


def turn_intent_onehot(intent_str):
    """Convert intent string to one-hot [left, straight, right]."""
    if intent_str == 'left':
        return [1.0, 0.0, 0.0]
    elif intent_str == 'right':
        return [0.0, 0.0, 1.0]
    else:
        return [0.0, 1.0, 0.0]


# -----------------------------------------------------------------------
#  Traffic light classification
# -----------------------------------------------------------------------
def traffic_light_class(vehicle):
    """
    Returns: 'red', 'yellow', 'green', or 'none'.
    Uses CARLA's built-in traffic light API.
    """
    if vehicle is None:
        return 'none'

    tl = vehicle.get_traffic_light()
    if tl is None:
        return 'none'

    state = tl.get_state()
    # carla.TrafficLightState: Red=0, Yellow=1, Green=2
    state_int = int(state)
    if state_int == 0:
        return 'red'
    elif state_int == 1:
        return 'yellow'
    elif state_int == 2:
        return 'green'
    return 'none'


def stop_required(vehicle, dist_threshold=15.0):
    """
    Returns True if there's a red/yellow light within dist_threshold meters.
    """
    if vehicle is None:
        return False

    tl = vehicle.get_traffic_light()
    if tl is None:
        return False

    state_int = int(tl.get_state())
    if state_int in (0, 1):  # Red or Yellow
        dist = vehicle.get_location().distance(tl.get_location())
        return dist < dist_threshold
    return False


# -----------------------------------------------------------------------
#  Full feature vector for one tick
# -----------------------------------------------------------------------
def compute_path_features(vehicle, ego_fwd, anchor_wp,
                          route_wps_with_cmds=None, route_idx=0,
                          route_total=1):
    """
    Compute full path feature dict for one tick.

    Returns dict with keys:
      hdg_delta_1..5, curvature_near/mid/far, dist_to_junction,
      turn_intent, route_progress, tl_class, stop_required, steer_smooth
    """
    features = {}

    # --- Heading deltas ---
    if route_wps_with_cmds and route_idx < len(route_wps_with_cmds):
        upcoming = route_wps_with_cmds[route_idx:route_idx + 5]
        deltas = heading_deltas(ego_fwd, upcoming, n=5)
    else:
        # Fallback: use anchor's next waypoints
        fallback_wps = []
        wp = anchor_wp
        for _ in range(5):
            nexts = wp.next(5.0)
            if nexts:
                wp = nexts[0]
                fallback_wps.append(wp)
        deltas = heading_deltas(ego_fwd, fallback_wps, n=5)

    for i, d in enumerate(deltas):
        features[f'hdg_delta_{i+1}'] = round(d, 4)

    # --- Curvature ---
    c_near, c_mid, c_far = curvature_features(anchor_wp)
    features['curvature_near'] = round(c_near, 5)
    features['curvature_mid'] = round(c_mid, 5)
    features['curvature_far'] = round(c_far, 5)

    # --- Distance to junction ---
    features['dist_to_junction'] = round(
        dist_to_next_junction(anchor_wp), 1)

    # --- Turn intent ---
    if route_wps_with_cmds:
        intent = turn_intent_from_route(route_wps_with_cmds, route_idx)
    else:
        intent = 'straight'
    features['turn_intent'] = intent

    # --- Route progress ---
    features['route_progress'] = round(
        route_idx / max(1, route_total), 4)

    # --- Traffic light ---
    features['tl_class'] = traffic_light_class(vehicle)
    features['stop_required'] = stop_required(vehicle)

    return features


def path_feature_vector(features_dict):
    """
    Convert feature dict to a 12-d numpy array for model input:
    [hdg_delta_1..5, curvature_near/mid/far, dist_junction_norm, intent_onehot(3)]
    """
    vec = []
    for i in range(1, 6):
        vec.append(features_dict.get(f'hdg_delta_{i}', 0.0))
    vec.append(features_dict.get('curvature_near', 0.0))
    vec.append(features_dict.get('curvature_mid', 0.0))
    vec.append(features_dict.get('curvature_far', 0.0))
    vec.append(features_dict.get('dist_to_junction', 100.0) / 100.0)  # normalize
    vec.extend(turn_intent_onehot(features_dict.get('turn_intent', 'straight')))
    return np.array(vec, dtype=np.float32)
