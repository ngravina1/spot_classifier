# 3D Spot Classification with PyTorch SE-ResNet

Deep learning for classifying 3D fluorescent spots in microscopy images using PyTorch. This project demonstrates implementation of 3D convolutional neural networks, including a custom SE-ResNet3D architecture with squeeze-and-excitation attention mechanisms.

## Project Overview

This repository extends the [zms2 image analysis pipeline](https://github.com/bschloma/zms2) (Eck, Moretti, and Schlomann, bioRxiv 2024) by implementing modern PyTorch-based deep learning models for automated spot classification in 3D microscopy data.

### Key Features

- **PyTorch Implementation**: Modern 3D CNNs for volumetric data classification
- **SE-ResNet3D Architecture**: Custom residual network with Squeeze-and-Excitation attention blocks
- **Interactive Notebook**: Complete model comparison with visualizations
- **Comprehensive Metrics**: Accuracy, precision, recall, AUC, confusion matrices, ROC curves

### Technologies

- **PyTorch** (2.0+): Deep learning framework
- **3D CNNs**: Volumetric convolutional networks for 3D image data
- **ResNet Architecture**: Residual connections for deeper networks
- **Squeeze-and-Excitation Blocks**: Attention mechanisms for channel-wise feature recalibration
- **CUDA**: GPU acceleration for training and inference

## Architecture Comparison

| Model | Parameters | Architecture Highlights |
|-------|-----------|------------------------|
| Simple CNN | 17,957 | 2 conv layers, baseline model |
| **SE-ResNet3D** | **900,001** | 3 residual blocks + SE attention |

## Repository Structure

```
spot_classifier/
├── zms2/                          # Core package
│   └── spots/
│       ├── classification_pytorch.py   # PyTorch SE-ResNet3D implementation
│       ├── trainer.py                  # Unified training interface
│       └── utils.py                    # Visualization utilities
│
├── notebooks/
│   ├── model_comparison.ipynb    # Interactive model comparison (main demo)
│   └── README.md                 # Notebook documentation
│
├── zms2_trainingData/            # Training data (not included in repo)
├── environment.yml               # Conda environment specification
├── setup.py                      # Package installation
└── README.md                     # This file
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd spot_classifier

# Create conda environment
conda env create -f environment.yml
conda activate spot_classifier

# Install package
pip install -e .
```

### Running the Comparison Notebook

The main demonstration is in the Jupyter notebook:

```bash
# Launch Jupyter Lab
jupyter lab

# Open notebooks/model_comparison.ipynb and run all cells
```

The notebook will:
- Load training data
- Train both Simple CNN and SE-ResNet3D models
- Display comprehensive comparison visualizations
- Show performance metrics

**Expected Runtime**: ~15-20 minutes with GPU (adjust epochs to 50 for faster demo)

### Training from Python

```python
from zms2.spots.trainer import create_trainer

# Create SE-ResNet3D trainer
trainer = create_trainer(
    'pytorch',
    base_channels=32,
    use_se=True,
    learning_rate=1e-4,
    batch_size=16,
    epochs=100
)

# Build and train
trainer.build_model()
history = trainer.train(X_train, y_train, X_val, y_val)

# Predict
predictions = trainer.predict(X_test)

# Save model
trainer.save('model.pt')
```

## Model Architecture Details

### SE-ResNet3D

```python
SE-ResNet3D(
  # Input: (B, 1, D, H, W) - 3D volumes

  # Initial convolution
  conv1: Conv3d(1, 32, kernel_size=3, padding=1)
  bn1: BatchNorm3d(32)

  # Residual blocks with SE attention
  block1: SEResidualBlock(32, 64)   # Downsample
  block2: SEResidualBlock(64, 128)  # Downsample
  block3: SEResidualBlock(128, 256) # Downsample

  # Classification head
  global_pool: AdaptiveAvgPool3d(1)
  fc: Linear(256, 1)

  # Total parameters: 900,001
)
```

**Key Features:**
- **Residual connections**: `output = F.relu(residual + x)`
- **SE blocks**: Channel attention via global pooling → FC → sigmoid
- **Downsampling**: Max pooling after each residual block
- **Regularization**: Batch normalization and dropout

## Credits

**Original zms2 Pipeline**:
- Repository: https://github.com/bschloma/zms2
- Paper: Eck, Moretti, and Schlomann, "Image analysis pipeline for MS2 reporters in large, dense tissues like zebrafish embryos", bioRxiv (2024)
- License: BSD 3-Clause

**PyTorch Implementation & SE-ResNet3D**:
- This repository adds PyTorch models and SE-ResNet3D architecture
- All PyTorch code in `zms2/spots/classification_pytorch.py`, `trainer.py`, and related files

## License

BSD 3-Clause License (inherited from zms2)

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support (recommended)
- 4GB+ GPU memory (or run on CPU with reduced batch size)
- See `environment.yml` for full dependencies
