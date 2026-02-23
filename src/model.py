import torch
import torch.nn as nn
import torchvision.models as models

class EndToEndDrivingModel(nn.Module):
    def __init__(self):
        super(EndToEndDrivingModel, self).__init__()
        
        # Vision backbone (Resnet18 for speed/simplicity)
        self.resnet = models.resnet18(pretrained=True)
        # Remove original fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # State processing branch (Speed)
        self.state_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined Prediction Heads
        # Takes ResNet features (512) + State features (64)
        combined_size = num_ftrs + 64
        
        self.fc_control = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # [throttle, steer, brake]
        )

    def forward(self, img, speed):
        # Image feature extraction
        img_features = self.resnet(img) # [batch, 512]
        
        # State feature extraction
        state_features = self.state_fc(speed) # [batch, 64]
        
        # Fusion
        combined = torch.cat((img_features, state_features), dim=1) # [batch, 576]
        
        # Regression outputs
        action = self.fc_control(combined)
        
        # Restrict outputs to their valid ranges
        # throttle: [0, 1], steer: [-1, 1], brake: [0, 1]
        throttle = torch.sigmoid(action[:, 0:1])
        steer = torch.tanh(action[:, 1:2])
        brake = torch.sigmoid(action[:, 2:3])
        
        return torch.cat((throttle, steer, brake), dim=1)

if __name__ == "__main__":
    # Test model shape
    model = EndToEndDrivingModel()
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_speed = torch.randn(2, 1)
    
    out = model(dummy_img, dummy_speed)
    print(f"Output shape: {out.shape}") # Should be [2, 3]
