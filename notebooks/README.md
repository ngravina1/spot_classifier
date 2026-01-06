# Model Comparison Notebook

Interactive demonstration of PyTorch 3D CNN architectures for spot classification.

## `model_comparison.ipynb`

**Complete comparison of Simple CNN vs SE-ResNet3D**

This notebook provides a step-by-step walkthrough of training, evaluating, and comparing two PyTorch deep learning models on the same 3D microscopy dataset.

### Key Functionality 

- **Data Loading**: Load and visualize training data distribution
- **Model Training**: Train both architectures with identical hyperparameters
- **Performance Metrics**: Comprehensive evaluation including:
  - Training/validation curves (loss and accuracy)
  - Confusion matrices
  - ROC curves with AUC scores
  - Classification reports
  - Inference speed benchmarks
- **Visual Comparison**: Side-by-side plots of all metrics:
  - **Accuracy**: Percentage of correct predictions
  - **Precision**: Of predicted positives, what fraction are actually positive (reduces false alarms)
  - **Recall**: Of actual positives, what fraction we correctly identified (reduces misses)
  - **AUC**: Area under ROC curve - overall discrimination ability (1.0 = perfect, 0.5 = random)


