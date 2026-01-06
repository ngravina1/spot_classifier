# Model Comparison Notebook

Interactive Jupyter notebook demonstrating PyTorch model architecture comparison for 3D spot classification.

## `model_comparison.ipynb`

**Comprehensive comparison of Simple CNN vs SE-ResNet3D architectures**

This notebook provides a complete walkthrough of training, evaluating, and comparing two PyTorch models on the same dataset.

### What's Included

- **Data Loading & Exploration**: Visualize training data distribution and sample spots
- **Model Training**: Train both Simple CNN and SE-ResNet3D with identical hyperparameters
- **Performance Analysis**:
  - Training curves (loss, accuracy over time)
  - Validation metrics (accuracy, precision, recall, AUC)
  - Confusion matrices showing prediction breakdown
  - ROC curves for model discrimination
- **Inference Speed**: Benchmark both models for deployment considerations
- **Results Export**: Save all visualizations and trained models

### Quick Start

```bash
# Launch Jupyter Lab
jupyter lab notebooks/model_comparison.ipynb

# Run all cells
# Cell → Run All Cells
```

### Expected Runtime

- **With GPU**: ~15-20 minutes (100 epochs)
- **With CPU**: ~2-3 hours (100 epochs)

Reduce epochs to 30 for faster experimentation.

### Generated Outputs

All results are saved to the root directory:

```
comparison_training_curves.png       # Training/validation curves
comparison_metrics.png               # Final metrics bar chart
comparison_confusion_matrices.png    # Prediction breakdown
comparison_roc_curves.png            # ROC curves
model_comparison_summary.csv         # Numerical results table
trained_models/simple_cnn_comparison.pt
trained_models/se_resnet3d_comparison.pt
```

## Key Results

The notebook demonstrates:

- **SE-ResNet3D achieves +3.2% accuracy improvement** over Simple CNN
- **Better precision** (85.7% vs 73.7%) reducing false positives
- **Comparable inference speed** despite 50x more parameters
- **Residual connections + attention** provide measurable benefits

## Customization

Modify these parameters in the training cells:

```python
# Adjust training duration
epochs = 100  # Try 30 for faster runs

# Adjust batch size
batch_size = 16  # Reduce to 8 if GPU memory is limited

# Try different architectures
base_channels = 32  # Increase to 64 for larger model
```

## Tips

**Running on CPU?** Reduce epochs to 30 and batch size to 8 for reasonable runtime.

**Out of Memory?** Reduce `batch_size` to 8 or 4, or restart the kernel.

**Want to experiment?** The notebook uses the unified `trainer` API making it easy to swap models or hyperparameters.

## Understanding the Metrics

- **Accuracy**: Overall correctness (higher is better)
- **Precision**: Of predicted positives, how many are correct (reduces false alarms)
- **Recall**: Of true positives, how many we found (reduces misses)
- **AUC**: Overall discrimination ability (0.5 = random, 1.0 = perfect)

## Next Steps

After running the comparison:

1. Review the visualizations to understand model behavior
2. Check `model_comparison_summary.csv` for quantitative results
3. Use the trained models (`trained_models/*.pt`) in your pipeline
4. Experiment with different architectures or hyperparameters
