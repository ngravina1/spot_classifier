# Example Scripts

Example implementations demonstrating PyTorch 3D CNN training and architecture comparison.

## `train_se_resnet3d.py`

Comprehensive training examples for SE-ResNet3D architecture.

**Demonstrates:**
- Basic training with train/validation split
- K-Fold cross-validation for robust evaluation
- Batch prediction on new data
- Model saving and loading

**Usage:**
```bash
python train_se_resnet3d.py
```

Note: Uses dummy data by default. Modify to load your own training data.

## `compare_architectures.py`

Architectural analysis and comparison tool.

**Compares:**
- Simple 3D CNN (baseline)
- SE-ResNet3D (advanced architecture)

**Outputs:**
- Parameter counts and model sizes
- Computational cost analysis
- Receptive field comparison
- Memory requirements
- Inference speed benchmarks

**Usage:**
```bash
python compare_architectures.py
```

## Quick Training Example

```python
from zms2.spots.trainer import create_trainer

# Create trainer
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

# Predict and save
predictions = trainer.predict(X_test)
trainer.save('model.pt')
```

## GPU Recommendations

- **Minimum**: 2GB GPU memory
- **Recommended**: 4GB+ for batch_size=16
- **CPU**: Supported but ~10x slower

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Tips

**Out of Memory?** Reduce batch size to 4 or 8.

**Slow Training?** Verify GPU usage and consider reducing epochs for experimentation.

**Poor Accuracy?** Ensure sufficient training data (>100 labeled examples) and check class balance.
