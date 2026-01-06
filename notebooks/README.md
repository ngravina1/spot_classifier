# Model Comparison Notebook

Interactive demonstration of PyTorch 3D CNN architectures for spot classification.

## `model_comparison.ipynb`

**Complete comparison of Simple CNN vs SE-ResNet3D**

This notebook provides a step-by-step walkthrough of training, evaluating, and comparing two PyTorch deep learning models on the same 3D microscopy dataset.

### What's Included

- **Data Loading**: Load and visualize training data distribution
- **Model Training**: Train both architectures with identical hyperparameters
- **Performance Metrics**: Comprehensive evaluation including:
  - Training/validation curves (loss and accuracy)
  - Confusion matrices
  - ROC curves with AUC scores
  - Classification reports
  - Inference speed benchmarks
- **Visual Comparison**: Side-by-side plots of all metrics

### Quick Start

```bash
# From repository root
jupyter lab notebooks/model_comparison.ipynb

# Run all cells: Cell → Run All Cells
```

### Runtime

- **With GPU**: ~15-20 minutes (50 epochs)
- **With CPU**: ~2-3 hours (50 epochs)

Adjust epochs in cells 9 and 12 for faster experimentation.

### Key Results

The notebook demonstrates:

- **SE-ResNet3D achieves significantly better accuracy** than Simple CNN
- **Better precision** reduces false positives
- **Comparable inference speed** despite 50x more parameters
- **Residual connections + attention** provide measurable benefits

### Customization

Modify training parameters:

```python
# Cell 9 and Cell 12: Adjust these values
epochs = 50        # Try 30 for quick demo, 100+ for final results
batch_size = 16    # Reduce to 8 if GPU memory limited
base_channels = 32 # Increase to 64 for larger model
```

### Tips

**GPU Not Available?** The notebook automatically detects and uses CPU. Reduce epochs to 30 and batch_size to 8 for reasonable runtime.

**Out of Memory?** Reduce `batch_size` to 8 or 4 in the trainer creation cells.

**Want to Experiment?** The unified `trainer` API makes it easy to swap models or adjust hyperparameters.

### Understanding the Metrics

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positives, what fraction are actually positive (reduces false alarms)
- **Recall**: Of actual positives, what fraction we correctly identified (reduces misses)
- **AUC**: Area under ROC curve - overall discrimination ability (1.0 = perfect, 0.5 = random)

## Next Steps

After running the comparison:

1. Review the visualizations to understand model behavior
2. Experiment with different hyperparameters
3. Try the trained models on your own data
4. Explore the source code in `zms2/spots/` to understand the implementations
