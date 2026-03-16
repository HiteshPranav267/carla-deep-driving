# End-to-End Autonomous Driving Research: CNN vs CNN-LSTM in CARLA

## Research Question
Can a CNN-LSTM architecture with regularization reduce phantom braking and adverse weather failures compared to a single-frame CNN baseline in CARLA?

This project directly addresses documented Tesla autopilot failure modes by comparing:
- **Model A**: Baseline CNN (single frame)
- **Model B**: CNN-LSTM (temporal sequence)

Both models are trained on identical data with identical hyperparameters - the only difference is temporal processing capability.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# 1. Collect training data (60,000 frames across 3 weather conditions)
python src/data_collector.py

# 2. Train both models separately
python src/train.py

# 3. Evaluate on reserved towns (Town03, Town05)
python src/evaluate.py

# 4. Generate attention visualizations
gradcam
python src/visualize.py

# 5. Run adversarial scenarios
tester
python src/scenario_tester.py
```

## Folder Structure
```
project/
├── src/
│   ├── model.py           # Model definitions (CNN vs CNN-LSTM)
│   ├── train.py           # Training pipeline
│   ├── data_collector.py  # Data collection from CARLA
│   ├── evaluate.py        # Model evaluation
│   ├── visualize.py       # GradCAM visualizations
│   └── scenario_tester.py  # Adversarial scenario testing
├── dataset/
│   ├── images/            # Collected training images
│   └── log.csv            # Training metadata
├── models/
│   ├── baseline_cnn.pth   # Trained CNN model
│   └── cnn_lstm.pth       # Trained CNN-LSTM model
├── results/
│   ├── evaluation_log.csv # Evaluation metrics
│   ├── scenario_results.csv # Scenario testing results
│   └── gradcam/           # Attention visualizations
├── requirements.txt
└── README.md
```

## Research Methodology
- **Training Data**: 60,000 frames from Town01, Town02, Town04 under clear, rainy, foggy conditions.
- **Model Architectures**: 
    - **Baseline CNN**: ResNet18 backbone for spatial feature extraction.
    - **CNN-LSTM**: ResNet18 + 2-layer LSTM for temporal sequence processing (Unit 3).
- **Bagging Ensemble (Bootstrap Aggregating)**: 
    - To reduce variance and eliminate "Phantom Braking," each model type is trained as an ensemble of **3 estimators**.
    - **Bootstrapping**: Each estimator is trained on a random sample of the dataset with replacement, ensuring diverse "viewpoints."
    - **Aggregation**: During inference, predictions (throttle, steer, brake) are averaged across all ensemble members to produce a stable control signal.
- **Regularization & Optimization**:
    - **Early Stopping**: Monitors validation loss with a patience of 7-10 epochs to prevent overfitting (Unit 2).
    - **Learning Rate Scheduling**: `ReduceLROnPlateau` factor of 0.5 to ensure convergence in local minima.
    - **Data Augmentation**: Random horizontal flips, color jitter, and Gaussian blur to improve out-of-distribution robustness.
- **Evaluation**: Town03 and Town05 reserved for testing only.
- **Metrics**: Route completion, collisions, phantom braking count, lane invasions, average speed.

## Syllabus Coverage
- **Unit 1**: CNN architecture, convolution, pooling, ResNet18 backbone
- **Unit 2**: Dropout, data augmentation, noise robustness, early stopping
- **Unit 3**: LSTM for temporal sequence processing
- **Unit 4**: Residual connections, batch normalization

## Results
All evaluation results are saved to `results/` directory for direct inclusion in research papers.

---
*Let's build!*