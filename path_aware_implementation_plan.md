# Path-Aware End-to-End Driving Policy — Implementation Plan

## Overview

Transform the current rule-assisted controller into a **learned path-aware policy** where the model handles steering, braking, and signal compliance — with path features providing turn intent and route context.

---

## Phase 1: Data Collection Upgrade (`data_collector_v2.py`)

### New CSV Schema (25 columns)

| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `timestamp` | float | existing | Frame timing |
| `frame_id` | str | existing | Image filename |
| `speed` | float | existing | Ego speed km/h |
| `speed_limit` | float | existing | Road limit |
| `throttle` | float | existing | Autopilot control |
| `brake` | float | existing | Autopilot control |
| [steer](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/evaluate.py#776-785) | float | existing | Autopilot control |
| `gap` | float | existing | Distance to lead |
| [status](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/evaluate.py#993-1026) | str | existing | Driving status |
| `lane_id` | int | existing | CARLA lane |
| `road_id` | int | existing | CARLA road |
| [weather](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/data_collector.py#101-110) | str | existing | Weather preset |
| `town` | str | existing | Map name |
| **`hdg_delta_1..5`** | float×5 | **NEW** | Heading angle to next 5 route waypoints (rad) |
| **`curvature_near`** | float | **NEW** | Road curvature 0-10m ahead |
| **`curvature_mid`** | float | **NEW** | Road curvature 10-30m ahead |
| **`curvature_far`** | float | **NEW** | Road curvature 30-60m ahead |
| **`dist_to_junction`** | float | **NEW** | Distance to next junction (m, capped 100) |
| **`turn_intent`** | str | **NEW** | `left`, `straight`, `right` for next maneuver |
| **`route_progress`** | float | **NEW** | Fraction of route completed |
| **`tl_class`** | str | **NEW** | `red`, `yellow`, `green`, `none` |
| **`stop_required`** | bool | **NEW** | Whether stop is needed at next controlled intersection |
| **`steer_smooth`** | float | **NEW** | EMA-filtered steer for smoother target |

### Route-Based Collection Protocol

```
For each episode:
  1. Pick random spawn point
  2. Pick random destination (far away, different road)
  3. Compute route via GlobalRoutePlanner
  4. Drive autopilot along route
  5. Each tick: compute path features relative to route
  6. Log all columns
  7. End episode on: route complete, collision, or timeout
```

### Balancing Rules

- After every 20k frames, check distribution
- If `turn_intent == straight` > 60% → force junction-focused spawns
- If `tl_class == red` < 5% → spawn in downtown areas
- 3 traffic seeds per town-weather combo

---

## Phase 2: Model Architecture ([model.py](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/model.py))

### PathAwareCNN

```
Inputs:
  - Image [B, 3, 224, 224]
  - Speed [B, 1]
  - Path features [B, 12]  (5 heading deltas + 3 curvatures + dist_junction + turn_intent_onehot(3))

Architecture:
  ResNet18 backbone → 512-d
  Speed MLP → 64-d
  Path MLP → 64-d
  Concat → BN → 640-d

Heads:
  Control → Linear(640, 256) → ReLU → Dropout → Linear(256, 3) → [throttle, steer, brake]
  TL Class → Linear(640, 128) → ReLU → Linear(128, 4) → softmax → [red, yellow, green, none]
  Intent → Linear(640, 64) → ReLU → Linear(64, 3) → softmax → [left, straight, right]
```

### PathAwareCNNLSTM

```
Same as above but:
  - Image sequence [B, 5, 3, 224, 224] through ResNet per-frame → LSTM
  - Path features from LAST frame only (current route state)
  - Same multi-task heads
```

### Path Feature Vector (12-d)

| Index | Feature | Range |
|-------|---------|-------|
| 0-4 | `hdg_delta_1..5` | [-π, π] rad |
| 5 | `curvature_near` | [0, 0.5] 1/m |
| 6 | `curvature_mid` | [0, 0.5] 1/m |
| 7 | `curvature_far` | [0, 0.5] 1/m |
| 8 | `dist_to_junction` | [0, 1] normalized by /100 |
| 9-11 | `turn_intent` one-hot | [0,1]×3 |

---

## Phase 3: Training ([train.py](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/train.py))

### Multi-Task Loss

```python
L_total = w_ctrl * L_control + w_tl * L_tl + w_intent * L_intent + w_smooth * L_steer_smooth

Where:
  L_control = MSE(pred_throttle, target_throttle) + MSE(pred_steer, target_steer) + MSE(pred_brake, target_brake)
  L_tl = CrossEntropy(pred_tl_class, target_tl_class)  # class-weighted
  L_intent = CrossEntropy(pred_intent, target_intent)    # class-weighted
  L_steer_smooth = MSE(pred_steer, target_steer_smooth)  # temporal consistency
```

### Curriculum Stages

| Stage | Epochs | What | Loss |
|-------|--------|------|------|
| 1 | 15 | Control only (freeze path/TL heads) | L_control only |
| 2 | 20 | Full multi-task | L_total (w_ctrl=1.0, w_tl=0.3, w_intent=0.3, w_smooth=0.1) |
| 3 | 10 | Hard-sample finetune | L_total on junction + red-light clips only |

### Class Weights

```python
tl_weights = [3.0, 5.0, 1.0, 1.0]  # red, yellow, green, none
intent_weights = [3.0, 1.0, 3.0]    # left, straight, right
```

---

## Phase 4: Evaluation ([evaluate.py](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/evaluate.py))

### Two Modes

| Mode | Steer Source | Brake Source | TL Handling |
|------|-------------|-------------|-------------|
| `pure_learned` | Model only | Model only | Model TL head only |
| `safety_overlay` | Model + lane assist | Model + rule brake | Model TL + rule backup |

### New Metrics

| Metric | How |
|--------|-----|
| `junction_success_rate` | Episodes with ≥1 junction: fraction with no crash near junction |
| `red_light_violations` | Count of running red (cross TL while red) |
| `steer_oscillation` | Variance of steer on straight segments (curvature < 0.01) |
| `tl_accuracy` | Classification accuracy of TL head vs ground truth |
| `intent_accuracy` | Classification accuracy of intent head |

### Benchmark Matrix

```
Models: [PathAwareCNN, PathAwareCNNLSTM, BaselineCNN, CNNLSTM]
Towns: [Town03, Town05]
Weather: [ClearNoon, WetNoon]
Episodes: 25 per combo
Modes: [pure_learned, safety_overlay]
```

---

## Phase 5: Ablations

| Ablation | Image | Speed | Path | Multi-task |
|----------|-------|-------|------|------------|
| A1: Baseline | ✓ | ✓ | ✗ | ✗ |
| A2: +Path | ✓ | ✓ | ✓ | ✗ |
| A3: +Multi-task | ✓ | ✓ | ✓ | ✓ |
| A4: CNN vs LSTM | Compare both architectures with A3 config | | | |
| A5: ±Safety | A3 with pure_learned vs safety_overlay | | | |

---

## Implementation Order

### Step 1: `data_collector_v2.py` (BUILD FIRST)
- [ ] Route-based episode structure
- [ ] Path feature computation per tick
- [ ] Extended CSV schema
- [ ] Balancing rules
- [ ] Quality gates

### Step 2: Collect 180k+ frames
- [ ] Town01: 60k (3 seeds × 4 weather × ~5k each)
- [ ] Town02: 60k
- [ ] Town04: 60k
- [ ] Distribution validation

### Step 3: [model.py](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/model.py) — Add PathAware architectures
- [ ] PathAwareCNN
- [ ] PathAwareCNNLSTM
- [ ] Keep existing models untouched for ablation baseline

### Step 4: [train.py](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/train.py) — Multi-task training pipeline
- [ ] PathAwareDataset with new CSV columns
- [ ] Multi-task loss function
- [ ] Curriculum training loop
- [ ] Hard-sample mining for Stage 3

### Step 5: [evaluate.py](file:///c:/Users/hites/OneDrive/Documents/clg%20stuff/Sem%206/DL/carla-deep-driving/src/evaluate.py) — Dual-mode evaluation
- [ ] `pure_learned` mode
- [ ] `safety_overlay` mode
- [ ] New metrics collection
- [ ] Failure taxonomy logging

### Step 6: Run experiments and ablations
- [ ] Train all model variants
- [ ] Run benchmark matrix
- [ ] Generate comparison tables

---

## File Map

```
carla-deep-driving/
├── src/
│   ├── data_collector.py        # Existing (keep for reference)
│   ├── data_collector_v2.py     # NEW: route-based, path-aware collection
│   ├── model.py                 # EXTEND: add PathAwareCNN, PathAwareCNNLSTM
│   ├── train.py                 # EXTEND: multi-task training
│   ├── evaluate.py              # EXTEND: dual-mode evaluation
│   └── path_features.py         # NEW: shared path feature computation
├── dataset/
│   ├── images/
│   └── log.csv                  # Will use new schema
└── models/
    ├── baseline_cnn_member_*.pth
    ├── cnn_lstm_member_*.pth
    ├── path_cnn_member_*.pth    # NEW
    └── path_lstm_member_*.pth   # NEW
```
