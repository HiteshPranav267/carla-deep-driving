import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import os
import csv
import numpy as np
import random

from model import EndToEndDrivingModel

class CarlaDataset(Dataset):
    def __init__(self, root_dir, transform=None, balance=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        csv_file = os.path.join(root_dir, "log.csv")
        img_dir = os.path.join(root_dir, "images")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found at {csv_file}")
            
        print("Indexing dataset and filtering non-existent images...")
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_name = row['frame']
                if os.path.exists(os.path.join(img_dir, frame_name)):
                    try:
                        speed = float(row["speed"])
                        throttle = float(row["throttle"])
                        steer = float(row["steer"])
                        brake = float(row["brake"])
                        
                        # Filter out NaNs/Infs
                        if any(np.isnan([speed, throttle, steer, brake])) or \
                           any(np.isinf([speed, throttle, steer, brake])):
                            continue
                            
                        # Sanitize/Clamp targets to valid ranges
                        throttle = np.clip(throttle, 0.0, 1.0)
                        steer = np.clip(steer, -1.0, 1.0)
                        brake = np.clip(brake, 0.0, 1.0)
                        
                        sample = {
                            "frame": frame_name,
                            "speed": speed,
                            "throttle": throttle,
                            "steer": steer,
                            "brake": brake
                        }
                        
                        # Dataset Balancing
                        if balance and abs(steer) < 0.05:
                            if random.random() < 0.3:
                                self.samples.append(sample)
                        else:
                            self.samples.append(sample)
                    except (ValueError, TypeError):
                        continue # Skip corrupted rows
                        
        print(f"Dataset loaded. Total samples after balancing: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, "images", sample["frame"])
        image = Image.open(img_path).convert("RGB")
        
        steer = sample["steer"]
        
        # Horizontal Flip Augmentation (Invert Steering)
        # ONLY during training (balance=True implies training mode here)
        if random.random() > 0.5:
             image = ImageOps.mirror(image)
             steer = -steer
        
        if self.transform:
            image = self.transform(image)
        
        # Normalize speed (approximate km/h to [0,1])
        speed = torch.tensor([sample["speed"] / 100.0], dtype=torch.float32) 
        actions = torch.tensor([sample["throttle"], steer, sample["brake"]], dtype=torch.float32)
        
        return image, speed, actions

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Robust Augmentations for 'Indian' Conditions (Dust/Chaos)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform (no jitter/flip inside dataset for val)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use project root paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found at {dataset_dir}. Please run data collection first.")
        return
        
    full_dataset = CarlaDataset(dataset_dir, transform=transform, balance=True)
    
    # Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Ensure validation subset uses val_transform (no jitter)
    val_dataset_base = CarlaDataset(dataset_dir, transform=val_transform, balance=False)
    val_loader = DataLoader(torch.utils.data.Subset(val_dataset_base, val_subset.indices), 
                            batch_size=32, shuffle=False)
    
    train_loader = DataLoader(train_dataset_subset, batch_size=32, shuffle=True)
    
    model = EndToEndDrivingModel().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    epochs = 30
    best_val_loss = float('inf')
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, speeds, targets) in enumerate(train_loader):
            images, speeds, targets = images.to(device), speeds.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, speeds)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}")
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for images, speeds, targets in val_loader:
                  images, speeds, targets = images.to(device), speeds.to(device), targets.to(device)
                  outputs = model(images, speeds)
                  loss = criterion(outputs, targets)
                  val_loss += loss.item() * images.size(0)
                  
        val_loss /= len(val_loader.dataset)
        print(f"EPOCH {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(project_root, "models", "best_model.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
