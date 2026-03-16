import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        # ResNet18 backbone with pretrained weights
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final classification layer

        # Batch normalization after fusion
        self.bn = nn.BatchNorm1d(512 + 64)

        # Speed processing MLP
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion and prediction heads
        self.fc_control = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [throttle, steer, brake]
        )

    def forward(self, img, speed):
        # Image feature extraction
        img_features = self.resnet(img)  # [batch, 512]

        # Speed feature extraction
        speed_features = self.speed_fc(speed)  # [batch, 64]

        # Fusion
        combined = torch.cat((img_features, speed_features), dim=1)  # [batch, 576]
        combined = self.bn(combined)

        # Regression outputs
        action = self.fc_control(combined)

        # Apply activation functions
        throttle = torch.sigmoid(action[:, 0:1])      # [0, 1]
        steer = torch.tanh(action[:, 1:2])            # [-1, 1]
        brake = torch.sigmoid(action[:, 2:3])         # [0, 1]

        return torch.cat((throttle, steer, brake), dim=1)

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()

        # ResNet18 backbone with pretrained weights
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final classification layer

        # Batch normalization after fusion (LSTM out 256 + speed 64)
        self.bn = nn.BatchNorm1d(256 + 64)

        # Speed processing MLP
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=512,      # ResNet features
            hidden_size=256,     # LSTM hidden size
            num_layers=2,        # 2 LSTM layers
            batch_first=True,
            dropout=0.3          # Dropout between LSTM layers
        )

        # Fusion and prediction heads
        self.fc_control = nn.Sequential(
            nn.Linear(256 + 64, 256),  # LSTM output + speed features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [throttle, steer, brake]
        )

    def forward(self, img_seq, speed):
        batch_size, seq_len, _, _, _ = img_seq.shape

        # Extract features from each frame in sequence
        img_features = []
        for t in range(seq_len):
            img_t = img_seq[:, t, :, :, :]
            features_t = self.resnet(img_t)  # [batch, 512]
            img_features.append(features_t)

        # Stack features: [batch, seq_len, 512]
        img_features = torch.stack(img_features, dim=1)

        # Process through LSTM
        lstm_out, _ = self.lstm(img_features)  # [batch, seq_len, 256]

        # Take the last time step output
        lstm_last = lstm_out[:, -1, :]  # [batch, 256]

        # Speed feature extraction
        speed_features = self.speed_fc(speed)  # [batch, 64]

        # Fusion
        combined = torch.cat((lstm_last, speed_features), dim=1)  # [batch, 320]
        combined = self.bn(combined)

        # Regression outputs
        action = self.fc_control(combined)

        # Apply activation functions
        throttle = torch.sigmoid(action[:, 0:1])      # [0, 1]
        steer = torch.tanh(action[:, 1:2])            # [-1, 1]
        brake = torch.sigmoid(action[:, 2:3])         # [0, 1]

        return torch.cat((throttle, steer, brake), dim=1)

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, img, speed):
        outputs = [model(img, speed) for model in self.models]
        # Average the predictions [batch, 3]
        stacked = torch.stack(outputs, dim=0)
        return torch.mean(stacked, dim=0)

# Factory function to create models by name
def create_model(model_name, n_estimators=1):
    if n_estimators > 1:
        models = []
        for _ in range(n_estimators):
            if model_name == 'baseline_cnn':
                models.append(BaselineCNN())
            elif model_name == 'cnn_lstm':
                models.append(CNNLSTM())
        return Ensemble(models)
    
    if model_name == 'baseline_cnn':
        return BaselineCNN()
    elif model_name == 'cnn_lstm':
        return CNNLSTM()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
    # Test model creation
    baseline = BaselineCNN()
    lstm_model = CNNLSTM()

    print("Baseline CNN:", baseline)
    print("CNN-LSTM:", lstm_model)