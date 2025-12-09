# PyTorch SE-ResNet3D Examples

This directory contains example scripts demonstrating the new PyTorch-based spot classification system.

## Files

### `train_se_resnet3d.py`
**Comprehensive training examples**

Demonstrates:
- Basic training with train/validation split
- K-Fold cross-validation
- Batch prediction on new data
- Model architecture comparison

**Usage:**
```bash
python train_se_resnet3d.py
```

**What it does:**
1. Creates dummy training data (replace with your real data)
2. Trains SE-ResNet3D models with various configurations
3. Saves trained models to `trained_models/`
4. Generates training curves and metrics
5. Runs inference on test data

### `compare_architectures.py`
**Architecture comparison tool**

Visualizes differences between:
- Old: TensorFlow/Keras simple 3D CNN
- New: PyTorch 3D SE-ResNet with attention

**Usage:**
```bash
python compare_architectures.py
```

**Outputs:**
- Side-by-side architecture diagrams
- Parameter counts
- Computational cost analysis
- Receptive field comparison
- Memory requirements
- Inference speed benchmarks (if GPU available)

## Quick Start

### 1. Install Dependencies

```bash
cd ../
pip install -r requirements_pytorch.txt
```

### 2. Run Example Training

```bash
cd examples
python train_se_resnet3d.py
```

This will train on dummy data. To use your real data, modify the script:

```python
# Replace this:
df = pd.DataFrame({
    'data': dummy_data,
    'manual_classification': dummy_labels
})

# With this:
df = pd.read_pickle('path/to/your/spots.pkl')
```

### 3. Compare Architectures

```bash
python compare_architectures.py
```

## Using with Real Data

### Training on Your Spots

```python
from zms2.spots.classification_pytorch import make_se_resnet3d, train_model
import pandas as pd
import torch

# Load your annotated spots
df = pd.read_pickle('annotated_spots.pkl')
# df must have columns: 'data' (list of 3D arrays) and 'manual_classification' (bool)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, history = train_model(
    df,
    learning_rate=1e-4,
    batch_size=16,
    epochs=100,
    device=device,
    verbose=True
)

# Save
from zms2.spots.classification_pytorch import save_model
save_model(model, 'my_spot_classifier.pt', history=history)
```

### Predicting on New Data

```python
from zms2.spots.classification_pytorch import run_batch_prediction
import pandas as pd

# Load unclassified spots
new_df = pd.read_pickle('spots_to_classify.pkl')

# Predict
results = run_batch_prediction(
    new_df,
    path_to_model='my_spot_classifier.pt',
    device='cuda',
    batch_size=32
)

# Filter by threshold
threshold = 0.5
positive_spots = results[results['prob'] > threshold]
print(f"Found {len(positive_spots)} positive spots")
```

## Expected Outputs

After running `train_se_resnet3d.py`, you'll find:

```
trained_models/
├── se_resnet3d_basic.pt          # Basic trained model
├── training_curves.png           # Loss/accuracy plots
└── kfold/
    ├── model_fold_1.pt
    ├── model_fold_2.pt
    ├── model_fold_3.pt
    ├── model_fold_4.pt
    ├── model_fold_5.pt
    ├── history_fold_1.pkl
    ├── history_fold_2.pkl
    ├── history_fold_3.pkl
    ├── history_fold_4.pkl
    └── history_fold_5.pkl
```

## GPU Requirements

- **Minimum**: 2GB GPU memory
- **Recommended**: 4GB+ for batch_size=16
- **CPU**: Works but ~10x slower

To check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
train_model(df, batch_size=4)
```

### Slow Training
```python
# Check you're using GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")
```

### Poor Accuracy
- Ensure you have enough training data (>100 manually labeled spots)
- Try longer training: `epochs=200`
- Increase model capacity: `base_channels=64`
- Check class balance: `print(df['manual_classification'].value_counts())`

## Next Steps

1. **Read the migration guide**: `../MIGRATION_GUIDE.md`
2. **Customize augmentation**: Edit `RandomRotation3D` class
3. **Tune hyperparameters**: Learning rate, batch size, architecture
4. **Visualize predictions**: Use napari to inspect results
5. **Ensemble models**: Combine K-Fold models for better accuracy

## Support

- **Documentation**: See `../MIGRATION_GUIDE.md`
- **Issues**: https://github.com/bschloma/zms2/issues
- **PyTorch Docs**: https://pytorch.org/docs/
