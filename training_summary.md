# 3-Model Comparison: Training Summary

This table shows the **average best validation metrics** across the 3 bagged estimators for each architecture.

| Model | Val MSE | Val Throttle MAE | Val Steer MAE | Val Brake MAE | Train Time (mins/est) |
|-------|---------|------------------|---------------|---------------|-----------------------|
| **baseline_cnn** | 0.01713 | 0.0835 | 0.0097 | 0.0449 | 320.9 |
| **cnn_gru** | 0.02692 | 0.1295 | 0.0147 | 0.0734 | 101.1 |
| **gru_only** | 0.03109 | 0.1451 | 0.0192 | 0.0900 | 176.5 |

> **Note**: To compute Steer R² and generate the final comparison CSV, run `python src/evaluate_offline.py`.
