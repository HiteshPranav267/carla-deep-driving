"""
train_v2.py — Strict 3-model comparison training pipeline.

Models trained (3 bagged estimators each):
  1. baseline_cnn  — ResNet18 + speed (single frame)
  2. cnn_gru       — ResNet18 → GRU + speed (5-frame sequence)
  3. gru_only      — Linear → GRU + speed (no CNN backbone)

Total: 3 models × 3 estimators = 9 .pth files

All models use IDENTICAL:
  - Data split (85/15, seed=42)
  - Augmentation (flip, jitter, blur, affine)
  - Loss (MSE on [throttle, steer, brake])
  - Optimizer (Adam, lr=1e-4, weight_decay=1e-5)
  - Scheduler (ReduceLROnPlateau)
  - Early stopping (patience=7)
  - Gradient clipping (max_norm=1.0)
  - Bagging (bootstrap sampling, 3 estimators)
  - Batch size (32 for CNN models, 16 for gru_only due to memory)

Run:
    python src/train_v2.py

Pause:
    Create STOP_TRAINING.flag in project root → saves checkpoint
    Delete flag and rerun → resumes from checkpoint
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import csv
import json
import numpy as np
import random
import time
import traceback
from collections import Counter

from model import create_model, count_parameters

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════
N_ESTIMATORS        = 3
BATCH_SIZE_DEFAULT  = 32
BATCH_SIZE_GRUONLY  = 16    # gru_only has ~77M params, needs smaller batch
MAX_EPOCHS          = 30
EARLY_STOP_PATIENCE = 7
LR                  = 1e-4
WEIGHT_DECAY        = 1e-5
GRAD_CLIP           = 1.0
NUM_WORKERS         = 2
VAL_FRACTION        = 0.15
SEQ_LEN             = 5

# Models to train — strict 3-model set
MODEL_LIST = ['baseline_cnn', 'cnn_gru', 'gru_only']


# ═══════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════
def _models_dir():
    return os.path.join(_PROJECT_ROOT, 'models')

def _best_path(mdir, model_name, est_i):
    return os.path.join(mdir, f'{model_name}_member_{est_i}.pth')

def _resume_path(mdir, model_name, est_i):
    return os.path.join(mdir, f'{model_name}_member_{est_i}_resume.pth')

def _stop_flag():
    return os.path.join(_PROJECT_ROOT, 'STOP_TRAINING.flag')

def _progress_path(mdir):
    return os.path.join(mdir, 'training_progress_v2.json')

def _log_path(mdir):
    return os.path.join(mdir, 'training_log.json')


# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════
class DrivingDataset(Dataset):
    """
    Unified dataset for all 3 models.
    Reads v2 CSV and returns (image_or_sequence, speed, control_target).
    For sequence models, returns 5 consecutive frames.
    """

    def __init__(self, csv_file, images_dir, model_type='baseline_cnn',
                 transform=None):
        self.images_dir = images_dir
        self.model_type = model_type
        self.transform  = transform
        self.is_temporal = model_type in ('cnn_gru', 'gru_only')

        # Parse all valid rows
        all_rows = []
        skipped  = 0

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = (row.get('frame_id') or '').strip()
                if not fid:
                    skipped += 1
                    continue
                if not os.path.exists(os.path.join(images_dir, fid)):
                    skipped += 1
                    continue
                all_rows.append(row)

        if skipped > 0:
            print(f"    Skipped {skipped} rows (missing image or empty)")

        self.data = all_rows

        # For temporal models: create non-overlapping 5-frame sequences
        if self.is_temporal:
            self._create_sequences()

        print(f"    [{model_type}] Loaded {len(self.data):,} samples")

    def _create_sequences(self):
        sequences = []
        current = []
        last_key = None

        for row in self.data:
            key = f"{row.get('town', '')}_{row.get('weather', '')}"
            ts = float(row.get('timestamp', 0) or 0)

            # Break on session change or time gap > 1s
            if last_key and key != last_key:
                current = []
            if len(current) > 0:
                prev_ts = float(current[-1].get('timestamp', 0) or 0)
                if abs(ts - prev_ts) > 1.0:
                    current = []

            current.append(row)
            last_key = key

            if len(current) == SEQ_LEN:
                sequences.append(current.copy())
                current = []  # Non-overlapping

        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_temporal:
            return self._get_sequence(idx)
        return self._get_single(idx)

    def _load_frame(self, row):
        """Load one frame → (image_tensor, speed, control)."""
        fid = row['frame_id']
        img = Image.open(os.path.join(self.images_dir, fid)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        speed = torch.tensor(
            [float(row.get('speed', 0) or 0) / 100.0],
            dtype=torch.float32)

        throttle = float(row.get('throttle', 0) or 0)
        steer    = float(row.get('steer_smooth', row.get('steer', 0)) or 0)
        brake    = float(row.get('brake', 0) or 0)
        control  = torch.tensor([throttle, steer, brake], dtype=torch.float32)

        return img, speed, control

    def _get_single(self, idx):
        row = self.data[idx]
        img, speed, control = self._load_frame(row)
        return img, speed, control

    def _get_sequence(self, idx):
        seq = self.data[idx]
        imgs = []
        for row in seq:
            img, speed, control = self._load_frame(row)
            imgs.append(img)

        # Stack images [5, C, H, W], return last frame's speed + control
        _, last_speed, last_control = self._load_frame(seq[-1])
        return torch.stack(imgs), last_speed, last_control


# ═══════════════════════════════════════════════════════════════
#  Transforms (identical for all models)
# ═══════════════════════════════════════════════════════════════
def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════
#  Training epoch
# ═══════════════════════════════════════════════════════════════
def _run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_samples  = 0
    throttle_ae = 0.0
    steer_ae    = 0.0
    brake_ae    = 0.0

    ctx = torch.no_grad() if not train else torch.enable_grad()

    with ctx:
        iterator = loader
        if tqdm is not None and train:
            iterator = tqdm(loader, desc="  Training",
                           leave=False, dynamic_ncols=True)

        for batch in iterator:
            imgs, speeds, controls = batch
            imgs     = imgs.to(device)
            speeds   = speeds.to(device)
            controls = controls.to(device)

            if train:
                optimizer.zero_grad()

            preds = model(imgs, speeds)
            loss = criterion(preds, controls)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

            # Per-channel MAE
            with torch.no_grad():
                ae = (preds - controls).abs()
                throttle_ae += ae[:, 0].sum().item()
                steer_ae    += ae[:, 1].sum().item()
                brake_ae    += ae[:, 2].sum().item()

            if tqdm is not None and train:
                iterator.set_postfix(loss=f"{loss.item():.5f}")

    n = max(1, n_samples)
    return {
        'loss':     total_loss / n,
        'thr_mae':  throttle_ae / n,
        'str_mae':  steer_ae / n,
        'brk_mae':  brake_ae / n,
    }


# ═══════════════════════════════════════════════════════════════
#  Train one model type (with bagging)
# ═══════════════════════════════════════════════════════════════
def train_model(model_name, csv_file, images_dir, models_dir,
                training_log, n_estimators=N_ESTIMATORS):
    """
    Train n_estimators bagged members for a single model type.
    All training parameters are IDENTICAL across models.
    """
    os.makedirs(models_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_temporal = model_name in ('cnn_gru', 'gru_only')
    batch_size  = BATCH_SIZE_GRUONLY if model_name == 'gru_only' else BATCH_SIZE_DEFAULT

    print(f"\n{'═'*60}")
    print(f"  Training: {model_name}")
    print(f"  Device:   {device}")
    print(f"  Temporal: {is_temporal}")
    print(f"  Batch:    {batch_size}")
    print(f"  Bagging:  {n_estimators} estimators × {MAX_EPOCHS} max epochs")

    # Create model to show parameter count
    model_tmp = create_model(model_name)
    print(f"  Params:   {count_parameters(model_tmp)/1e6:.1f}M")
    del model_tmp
    print(f"{'═'*60}\n")

    # ── Build datasets ──
    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    print("  Loading training dataset...")
    full_ds = DrivingDataset(csv_file, images_dir, model_type=model_name,
                             transform=train_tf)

    print("  Loading validation dataset...")
    val_ds = DrivingDataset(csv_file, images_dir, model_type=model_name,
                            transform=val_tf)

    if len(full_ds) < 10:
        print(f"  ✗ Not enough data ({len(full_ds)}). Skipping.")
        return

    # ── Identical split for all models ──
    n = len(full_ds)
    n_val = max(1, int(VAL_FRACTION * n))
    n_train = n - n_val
    print(f"  Split: {n_train:,} train / {n_val:,} val")

    # Use same seed for split reproducibility
    gen = torch.Generator().manual_seed(SEED)
    train_split, val_split = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=gen)

    val_loader = DataLoader(val_split, batch_size=batch_size,
                            shuffle=False, num_workers=NUM_WORKERS)

    criterion = nn.MSELoss()

    # ── Progress tracking ──
    prog_path = _progress_path(models_dir)
    progress = {}
    if os.path.exists(prog_path):
        try:
            with open(prog_path, 'r') as f:
                progress = json.load(f)
        except Exception:
            pass
    completed = progress.get(model_name, {}).get('completed', [])

    for est_i in range(n_estimators):
        if est_i in completed:
            print(f"\n  Estimator {est_i+1}/{n_estimators} already done. Skipping.")
            continue

        print(f"\n{'─'*50}")
        print(f"  Estimator {est_i+1}/{n_estimators}: {model_name}")
        print(f"{'─'*50}")

        save_path   = _best_path(models_dir, model_name, est_i)
        resume_ckpt = _resume_path(models_dir, model_name, est_i)

        if os.path.exists(save_path) and not os.path.exists(resume_ckpt):
            print(f"  Already trained ({save_path}). Skipping.")
            if est_i not in completed:
                completed.append(est_i)
                progress[model_name] = {'completed': completed}
                with open(prog_path, 'w') as f:
                    json.dump(progress, f, indent=2)
            continue

        model = create_model(model_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4)

        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 0

        # Resume from checkpoint
        if os.path.exists(resume_ckpt):
            print(f"  Resuming from checkpoint...")
            ckpt = torch.load(resume_ckpt, map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'scheduler_state' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
                except Exception:
                    pass
            best_val_loss    = ckpt.get('best_val_loss', best_val_loss)
            patience_counter = ckpt.get('patience_counter', 0)
            start_epoch      = ckpt.get('epoch', 0) + 1
            print(f"  → epoch={start_epoch}, best={best_val_loss:.5f}")

        # ── Bootstrap sampling for bagging ──
        rng = np.random.RandomState(SEED + est_i)  # Different bootstrap per estimator
        indices = rng.choice(len(train_split), len(train_split), replace=True)
        boot_subset = Subset(train_split, indices.tolist())
        train_loader = DataLoader(boot_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

        # ── Training loop ──
        log_key = f"{model_name}_est{est_i}"
        if log_key not in training_log:
            training_log[log_key] = []

        for epoch in range(start_epoch, MAX_EPOCHS):
            # Check pause flag
            if os.path.exists(_stop_flag()):
                print("\n  ⏸  STOP_TRAINING.flag detected! Saving checkpoint.")
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': max(0, epoch - 1),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                }, resume_ckpt)
                _save_log(models_dir, training_log)
                print("  Delete STOP_TRAINING.flag and rerun to resume.")
                return

            t0 = time.time()

            # Train
            t_metrics = _run_epoch(model, train_loader, optimizer,
                                   criterion, device, train=True)
            # Validate
            v_metrics = _run_epoch(model, val_loader, None,
                                   criterion, device, train=False)

            scheduler.step(v_metrics['loss'])
            elapsed = time.time() - t0

            # Log
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': round(t_metrics['loss'], 6),
                'val_loss': round(v_metrics['loss'], 6),
                'val_thr_mae': round(v_metrics['thr_mae'], 5),
                'val_str_mae': round(v_metrics['str_mae'], 5),
                'val_brk_mae': round(v_metrics['brk_mae'], 5),
                'lr': optimizer.param_groups[0]['lr'],
                'time_s': round(elapsed, 1),
            }
            training_log[log_key].append(epoch_log)

            line = (f"  Ep {epoch+1:2d}/{MAX_EPOCHS} │ "
                    f"T:{t_metrics['loss']:.5f} V:{v_metrics['loss']:.5f} │ "
                    f"thr:{v_metrics['thr_mae']:.4f} "
                    f"str:{v_metrics['str_mae']:.4f} "
                    f"brk:{v_metrics['brk_mae']:.4f} │ "
                    f"{elapsed:.0f}s")

            if v_metrics['loss'] < best_val_loss:
                best_val_loss = v_metrics['loss']
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                line += " ★"
            else:
                patience_counter += 1
                line += f" (pat {patience_counter}/{EARLY_STOP_PATIENCE})"

            print(line)

            # Save resume checkpoint
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
            }, resume_ckpt)

            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch+1}.")
                break

        # Mark complete
        if os.path.exists(resume_ckpt):
            os.remove(resume_ckpt)
        completed.append(est_i)
        progress[model_name] = {'completed': completed}
        with open(prog_path, 'w') as f:
            json.dump(progress, f, indent=2)
        _save_log(models_dir, training_log)
        print(f"  ✓ Estimator {est_i+1} saved: {save_path} "
              f"(best_val={best_val_loss:.5f})")

    print(f"\n  All {n_estimators} estimators done for {model_name}.")


def _save_log(models_dir, training_log):
    """Persist training log to JSON."""
    path = _log_path(models_dir)
    with open(path, 'w') as f:
        json.dump(training_log, f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  Dataset stats (data readiness check)
# ═══════════════════════════════════════════════════════════════
def print_dataset_stats(csv_file):
    """Print class distribution stats — data readiness gate."""
    status_counts  = Counter()
    town_counts    = Counter()
    weather_counts = Counter()
    steer_turning  = 0
    speed_sum      = 0.0
    total          = 0

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            status_counts[(row.get('status') or 'Unknown').strip()] += 1
            town_counts[(row.get('town') or 'Unknown').strip()] += 1
            weather_counts[(row.get('weather') or 'Unknown').strip()] += 1
            s = abs(float(row.get('steer', 0) or 0))
            if s > 0.01:
                steer_turning += 1
            speed_sum += float(row.get('speed', 0) or 0)

    mean_speed = speed_sum / max(1, total)

    print(f"\n{'═'*60}")
    print(f"  DATA READINESS CHECK ({total:,} rows)")
    print(f"{'═'*60}")

    # D1: Row count
    gate_rows = total >= 10000
    print(f"\n  D1 Row count:       {total:,} {'✅' if gate_rows else '❌ FAIL (need ≥10k)'}")

    # D4: Towns
    n_towns = len(town_counts)
    gate_towns = n_towns >= 2
    print(f"  D4 Towns:           {n_towns} {'✅' if gate_towns else '❌ FAIL (need ≥2)'}")
    for t, c in sorted(town_counts.items(), key=lambda x: -x[1]):
        print(f"     {t:15s}: {c:8,} ({c/total*100:5.1f}%)")

    # D5: Weather
    n_weather = len(weather_counts)
    gate_weather = n_weather >= 2
    print(f"  D5 Weather presets: {n_weather} {'✅' if gate_weather else '❌ FAIL (need ≥2)'}")
    for w, c in sorted(weather_counts.items(), key=lambda x: -x[1]):
        print(f"     {w:15s}: {c:8,} ({c/total*100:5.1f}%)")

    # D6: Speed
    gate_speed = mean_speed > 5
    print(f"  D6 Mean speed:      {mean_speed:.1f} km/h {'✅' if gate_speed else '❌ FAIL (need >5)'}")

    # D7: Steer diversity
    steer_pct = steer_turning / max(1, total) * 100
    gate_steer = steer_pct >= 5
    print(f"  D7 Turning rows:    {steer_pct:.1f}% {'✅' if gate_steer else '❌ FAIL (need ≥5%)'}")

    # Status distribution
    print(f"\n  Status distribution:")
    for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"     {s:20s}: {c:8,} ({c/total*100:5.1f}%)")

    all_pass = all([gate_rows, gate_towns, gate_weather, gate_speed, gate_steer])
    print(f"\n  {'✅ ALL GATES PASSED — ready to train!' if all_pass else '❌ SOME GATES FAILED — review data before training!'}")
    print(f"{'═'*60}\n")

    return all_pass


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    dataset_dir = os.path.join(_PROJECT_ROOT, 'dataset_v2')
    images_dir  = os.path.join(dataset_dir, 'images')
    csv_file    = os.path.join(dataset_dir, 'log.csv')
    models_dir  = os.path.join(_PROJECT_ROOT, 'models')

    if not os.path.exists(csv_file):
        print(f"Dataset not found at {csv_file}")
        print("Run data_collector_v2.py first.")
        return

    print("═" * 60)
    print("  3-Model Comparison Study — Training Pipeline")
    print("  baseline_cnn × cnn_gru × gru_only")
    print("  3 estimators each = 9 models total")
    print("═" * 60)
    print(f"\n  Dataset: {csv_file}")
    print(f"  Models:  {models_dir}")
    print(f"  Pause:   Create '{os.path.basename(_stop_flag())}' to pause")

    # ── Data readiness gate ──
    gates_ok = print_dataset_stats(csv_file)
    if not gates_ok:
        print("  ⚠  Data readiness gates not met. Training anyway...")

    # Check pause flag
    if os.path.exists(_stop_flag()):
        print(f"\n  ⚠ STOP_TRAINING.flag exists! Delete it to begin.")
        return

    # ── Training log ──
    training_log = {}
    log_file = _log_path(models_dir)
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                training_log = json.load(f)
        except Exception:
            pass

    # ── Train all 3 models ──
    for model_name in MODEL_LIST:
        if os.path.exists(_stop_flag()):
            print(f"\n  ⏸ Paused before {model_name}.")
            return

        try:
            train_model(model_name, csv_file, images_dir, models_dir,
                       training_log)
        except Exception as e:
            print(f"\n  ✗ Error training {model_name}: {e}")
            traceback.print_exc()
            continue

    # ── Save final log ──
    _save_log(models_dir, training_log)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Models saved in: {models_dir}")
    print(f"{'═'*60}")

    if os.path.isdir(models_dir):
        models = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])
        for m in models:
            size = os.path.getsize(os.path.join(models_dir, m)) / 1e6
            print(f"    {m} ({size:.1f} MB)")

    # Print per-model best val losses
    print(f"\n  Summary:")
    for model_name in MODEL_LIST:
        losses = []
        for est_i in range(N_ESTIMATORS):
            key = f"{model_name}_est{est_i}"
            if key in training_log and training_log[key]:
                best = min(e['val_loss'] for e in training_log[key])
                losses.append(best)
        if losses:
            mean_loss = np.mean(losses)
            std_loss  = np.std(losses)
            print(f"    {model_name:15s}: val_loss = {mean_loss:.5f} ± {std_loss:.5f}")


if __name__ == "__main__":
    main()
