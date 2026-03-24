"""
evaluate_offline.py — Generate offline metrics table for the 3-model comparison.
Runs the validation split to compute Val MSE, MAEs, and Steer R².
Outputs results to results/comparison_metrics.csv.
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time

from model import create_model, count_parameters
# We reuse DrivingDataset and get_val_transforms from train_v2
from train_v2 import DrivingDataset, get_val_transforms, VAL_FRACTION, SEED, BATCH_SIZE_DEFAULT

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def evaluate_offline():
    dataset_dir = os.path.join(_PROJECT_ROOT, 'dataset_v2')
    images_dir  = os.path.join(dataset_dir, 'images')
    csv_file    = os.path.join(dataset_dir, 'log.csv')
    
    models_dir = os.path.join(_PROJECT_ROOT, 'models')
    results_dir = os.path.join(_PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Offline Evaluation on {device}...")
    
    val_tf = get_val_transforms()
    criterion = nn.MSELoss()
    
    models_to_eval = ['baseline_cnn', 'cnn_gru', 'gru_only']
    results = []

    for model_name in models_to_eval:
        print(f"\nEvaluating {model_name}...")
        
        # Determine dataset (to match training)
        is_temporal = model_name in ('cnn_gru', 'gru_only')
        batch_size = 16 if model_name == 'gru_only' else BATCH_SIZE_DEFAULT
        
        ds = DrivingDataset(csv_file, images_dir, model_type=model_name, transform=val_tf)
        n = len(ds)
        n_val = max(1, int(VAL_FRACTION * n))
        n_train = n - n_val
        
        gen = torch.Generator().manual_seed(SEED)
        _, val_split = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=2)

        # Create ensemble model
        members = []
        for est_i in range(3):
            ckpt_path = os.path.join(models_dir, f'{model_name}_member_{est_i}.pth')
            if not os.path.exists(ckpt_path):
                print(f"  Missing {ckpt_path}, skipping estimator.")
                continue
            m = create_model(model_name)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            m.to(device)
            m.eval()
            members.append(m)
        
        if not members:
            print(f"  No estimators found for {model_name}. Skipping.")
            continue
            
        print(f"  Loaded {len(members)} estimators.")
        param_count = count_parameters(members[0]) / 1e6
        
        # Evaluate
        total_loss = 0.0
        n_samples = 0
        throttle_ae = 0.0
        steer_ae = 0.0
        brake_ae = 0.0
        
        all_steer_true = []
        all_steer_pred = []
        
        t0 = time.time()
        
        with torch.no_grad():
            for imgs, speeds, controls in val_loader:
                imgs = imgs.to(device)
                speeds = speeds.to(device)
                controls = controls.to(device)
                
                # Ensemble prediction
                preds = []
                for m in members:
                    preds.append(m(imgs, speeds))
                ensemble_pred = torch.mean(torch.stack(preds, dim=0), dim=0)
                
                loss = criterion(ensemble_pred, controls)
                
                bs = imgs.size(0)
                total_loss += loss.item() * bs
                n_samples += bs
                
                ae = (ensemble_pred - controls).abs()
                throttle_ae += ae[:, 0].sum().item()
                steer_ae    += ae[:, 1].sum().item()
                brake_ae    += ae[:, 2].sum().item()
                
                all_steer_true.extend(controls[:, 1].cpu().numpy())
                all_steer_pred.extend(ensemble_pred[:, 1].cpu().numpy())

        total_time = time.time() - t0
        
        mse = total_loss / max(1, n_samples)
        t_mae = throttle_ae / max(1, n_samples)
        s_mae = steer_ae / max(1, n_samples)
        b_mae = brake_ae / max(1, n_samples)
        r2 = compute_r2(np.array(all_steer_true), np.array(all_steer_pred))

        # Get training time from log if available
        train_time = "Unknown"
        try:
            import json
            log_path = os.path.join(models_dir, 'training_log.json')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                
                # sum time for all estimators
                Total_time = 0
                for est_i in range(3):
                    key = f"{model_name}_est{est_i}"
                    if key in log_data:
                        Total_time += sum(e.get('time_s', 0) for e in log_data[key])
                if Total_time > 0:
                    mins = Total_time / 60
                    train_time = f"{mins:.0f}m"
        except Exception:
            pass
        
        results.append({
            'Model': model_name,
            'Val MSE': mse,
            'Throttle MAE': t_mae,
            'Steer MAE': s_mae,
            'Brake MAE': b_mae,
            'Steer R²': r2,
            'Params (M)': round(param_count, 1),
            'Train Time': train_time
        })
        
        print(f"  MSE: {mse:.5f} | Steer R²: {r2:.3f}")

    # Output to CSV
    df = pd.DataFrame(results)
    out_csv = os.path.join(results_dir, 'comparison_metrics.csv')
    df.to_csv(out_csv, index=False)
    
    print(f"\nResults saved to {out_csv}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    evaluate_offline()
