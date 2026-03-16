import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import os
import csv
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
from model import create_model
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

class CarlaDataset(Dataset):
    def __init__(self, csv_file, images_dir, model_type='baseline_cnn', transform=None):
        self.csv_file = csv_file
        self.images_dir = images_dir
        self.model_type = model_type
        self.transform = transform
        self.data = []
        self.skipped_rows = 0
        self.skipped_empty_rows = 0
        self.skipped_short_rows = 0

        # Load CSV data
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            self.columns = next(reader)  # Read header
            expected_cols = len(self.columns)

            for row in reader:
                if not row or all(not field.strip() for field in row):
                    self.skipped_rows += 1
                    self.skipped_empty_rows += 1
                    continue
                if len(row) < expected_cols:
                    self.skipped_rows += 1
                    self.skipped_short_rows += 1
                    continue
                self.data.append(row)

        # For CNN-LSTM, we need to group frames into sequences
        if model_type == 'cnn_lstm':
            self._create_sequences()

        if self.skipped_rows > 0:
            print(
                f"[{model_type}] Loaded {len(self.data)} records; "
                f"skipped {self.skipped_rows} malformed rows "
                f"(empty={self.skipped_empty_rows}, short={self.skipped_short_rows})."
            )

    def _create_sequences(self):
        """Group frames into sequences of 5 consecutive frames from same session"""
        sequences = []
        current_sequence = []
        last_id = None
        last_time = None
        skipped_sequence_rows = 0

        for row in self.data:
            try:
                town = row[12]
                weather = row[11]
                time = float(row[0])
            except (ValueError, IndexError):
                skipped_sequence_rows += 1
                continue
            current_id = f"{town}_{weather}"

            # Detect session break using town/weather OR a large time jump
            is_new_session = (last_id is not None and current_id != last_id) or \
                             (last_time is not None and (time - last_time) > 1.0)

            if is_new_session:
                current_sequence = []

            current_sequence.append(row)
            last_id = current_id
            last_time = time
            
            if len(current_sequence) == 5:
                sequences.append(current_sequence.copy())
                current_sequence = [] # Non-overlapping as requested

        if skipped_sequence_rows > 0:
            print(f"[cnn_lstm] Skipped {skipped_sequence_rows} rows during sequence parsing.")

        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.model_type == 'baseline_cnn':
            return self._get_single_frame(idx)
        else:  # cnn_lstm
            return self._get_sequence(idx)

    def _get_single_frame(self, idx):
        row = self.data[idx]
        frame_name = row[1]
        speed = float(row[2])
        throttle = float(row[4])
        brake = float(row[5])
        steer = float(row[6])

        # Load image
        img_path = os.path.join(self.images_dir, frame_name)
        image = Image.open(img_path).convert('RGB')

        # Apply augmentations (only during training)
        if self.transform:
            image = self.transform(image)

        # Normalize speed to [0, 1]
        speed_tensor = torch.tensor([speed / 100.0], dtype=torch.float32)

        # Create target tensor
        targets = torch.tensor([throttle, steer, brake], dtype=torch.float32)

        return image, speed_tensor, targets

    def _get_sequence(self, idx):
        sequence = self.data[idx]

        images = []
        speeds = []
        targets = []

        for row in sequence:
            frame_name = row[1]
            speed = float(row[2])
            throttle = float(row[4])
            brake = float(row[5])
            steer = float(row[6])

            # Load image
            img_path = os.path.join(self.images_dir, frame_name)
            image = Image.open(img_path).convert('RGB')

            # Apply augmentations (only during training)
            if self.transform:
                image = self.transform(image)

            images.append(image)
            speeds.append(speed / 100.0)  # Normalize speed
            targets.append([throttle, steer, brake])

        # Stack images: [5, 224, 224, 3]
        images_tensor = torch.stack(images, dim=0)  # [seq_len, C, H, W]
        speeds_tensor = torch.tensor(speeds, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
        targets_tensor = torch.tensor(targets, dtype=torch.float32)  # [seq_len, 3]

        return images_tensor, speeds_tensor[-1], targets_tensor[-1]  # Return last frame's speed and target

def get_train_transforms():
    """Data augmentations for training (Unit 2 coverage)"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """Transforms for validation (no augmentations)"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def _load_progress(models_dir):
    path = os.path.join(models_dir, 'training_progress.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def _save_progress(models_dir, progress):
    path = os.path.join(models_dir, 'training_progress.json')
    with open(path, 'w') as f:
        json.dump(progress, f, indent=2)

def train_bagging_ensemble(model_name, dataset, val_dataset, n_estimators=3, epochs=30):
    print(f"\n===== TRAINING BAGGING ENSEMBLE: {model_name} ({n_estimators} estimators) =====")
    
    # Portable path for saving models
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_path, 'models')
    os.makedirs(models_dir, exist_ok=True)

    progress = _load_progress(models_dir)
    completed = progress.get(model_name, {}).get('completed', [])

    for i in range(n_estimators):
        if i in completed:
            print(f"\n--- Estimator {i+1}/{n_estimators} already complete, skipping. ---")
            continue

        print(f"\n--- Training Estimator {i+1}/{n_estimators} ---")
        
        # 1. Bootstrap sampling: sample indices WITH replacement
        indices = np.random.choice(len(dataset), len(dataset), replace=True)
        bootstrapped_subset = torch.utils.data.Subset(dataset, indices)
        
        train_loader = DataLoader(bootstrapped_subset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        # 2. Create and train individual model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(model_name).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        early_stop_patience = 7 # Slightly aggressive for bagging
        early_stop_counter = 0
        start_epoch = 0

        # Resume from epoch checkpoint if one exists
        resume_path = os.path.join(models_dir, f'{model_name}_member_{i}_resume.pth')
        if os.path.exists(resume_path):
            print(f"  Resuming Est {i+1} from checkpoint {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt['best_val_loss']
            early_stop_counter = ckpt['early_stop_counter']
            print(f"  Resuming from epoch {start_epoch + 1}/{epochs} | best_val={best_val_loss:.5f}")

        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0.0
            train_iter = train_loader
            if tqdm is not None:
                train_iter = tqdm(
                    train_loader,
                    desc=f"Est {i+1}/{n_estimators} | Epoch {epoch+1}/{epochs} | Train",
                    leave=False,
                    dynamic_ncols=True,
                )

            for batch_idx, (inputs, speeds, targets) in enumerate(train_iter):
                inputs, speeds, targets = inputs.to(device), speeds.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, speeds)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

                if tqdm is not None:
                    train_iter.set_postfix(loss=f"{loss.item():.5f}")
                else:
                    print(
                        f"\rEst {i+1}/{n_estimators} | Epoch {epoch+1}/{epochs} | "
                        f"Train batch {batch_idx+1}/{len(train_loader)}",
                        end="",
                        flush=True,
                    )

            if tqdm is None:
                print()

            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_iter = val_loader
                if tqdm is not None:
                    val_iter = tqdm(
                        val_loader,
                        desc=f"Est {i+1}/{n_estimators} | Epoch {epoch+1}/{epochs} | Val",
                        leave=False,
                        dynamic_ncols=True,
                    )

                for batch_idx, (inputs, speeds, targets) in enumerate(val_iter):
                    inputs, speeds, targets = inputs.to(device), speeds.to(device), targets.to(device)
                    outputs = model(inputs, speeds)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                    if tqdm is not None:
                        val_iter.set_postfix(loss=f"{loss.item():.5f}")
                    else:
                        print(
                            f"\rEst {i+1}/{n_estimators} | Epoch {epoch+1}/{epochs} | "
                            f"Val batch {batch_idx+1}/{len(val_loader)}",
                            end="",
                            flush=True,
                        )

                if tqdm is None:
                    print()

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            print(f"Est {i+1} | Ep {epoch+1} | T: {train_loss:.5f} | V: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_path = os.path.join(models_dir, f'{model_name}_member_{i}.pth')
                torch.save(model.state_dict(), save_path)
            else:
                early_stop_counter += 1

            # Save epoch resume checkpoint so training can be interrupted and resumed.
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stop_counter,
            }, resume_path)

            if early_stop_counter >= early_stop_patience:
                break

        # Mark this estimator as complete and clean up the resume checkpoint.
        completed.append(i)
        progress[model_name] = {'completed': completed}
        _save_progress(models_dir, progress)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        print(f"  Estimator {i+1} complete. Progress saved.")

    print(f"Finished Bagging Ensemble for {model_name}")

def main():
    # Portable path resolution
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_path, 'dataset')
    images_dir = os.path.join(dataset_dir, 'images')
    csv_file = os.path.join(dataset_dir, 'log.csv')

    if not os.path.exists(csv_file):
        print(f"Dataset not found at {csv_file}. Please run data_collector.py first.")
        return

    print("Loading dataset for Bagging Ensemble...")

    # Transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    # Create full datasets
    baseline_dataset = CarlaDataset(csv_file, images_dir, model_type='baseline_cnn', transform=train_transform)
    lstm_dataset = CarlaDataset(csv_file, images_dir, model_type='cnn_lstm', transform=train_transform)

    if len(baseline_dataset) < 2 or len(lstm_dataset) < 2:
        print("Not enough data after filtering to create train/validation splits.")
        return

    # Split independently per dataset because cnn_lstm length differs after sequence grouping.
    baseline_train_size = int(0.8 * len(baseline_dataset))
    baseline_val_size = len(baseline_dataset) - baseline_train_size
    lstm_train_size = int(0.8 * len(lstm_dataset))
    lstm_val_size = len(lstm_dataset) - lstm_train_size

    # Ensure both splits have at least one sample.
    if baseline_val_size == 0:
        baseline_train_size -= 1
        baseline_val_size = 1
    if lstm_val_size == 0:
        lstm_train_size -= 1
        lstm_val_size = 1

    # We use fixed splits for validation, but bootstrapping for training.
    _, baseline_val = torch.utils.data.random_split(
        baseline_dataset,
        [baseline_train_size, baseline_val_size]
    )
    _, lstm_val = torch.utils.data.random_split(
        lstm_dataset,
        [lstm_train_size, lstm_val_size]
    )

    # Train ensembles
    train_bagging_ensemble('baseline_cnn', baseline_dataset, baseline_val, n_estimators=3)
    train_bagging_ensemble('cnn_lstm', lstm_dataset, lstm_val, n_estimators=3)

if __name__ == "__main__":
    main()