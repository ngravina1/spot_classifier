"""Example script for training and using the PyTorch SE-ResNet3D spot classifier.

This script demonstrates:
1. Loading spot data from a DataFrame
2. Training a SE-ResNet3D model
3. Evaluating the model
4. Running predictions on new data
5. K-Fold cross-validation

Usage:
    python train_se_resnet3d.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zms2.spots.classification_pytorch import (
    make_se_resnet3d,
    train_model,
    train_model_kfold,
    run_batch_prediction,
    save_model,
    load_model
)


def example_basic_training():
    """Example 1: Basic training with train/validation split."""
    print("="*60)
    print("Example 1: Basic Training")
    print("="*60)

    # Load your spots DataFrame
    # df should have columns: 'data' (list of 3D numpy arrays) and 'manual_classification' (bool)
    # For this example, we'll create dummy data
    print("\nCreating dummy training data...")
    n_spots = 200
    dummy_data = []
    dummy_labels = []

    for i in range(n_spots):
        # Create random 9x11x11 volumes
        volume = np.random.rand(9, 11, 11).astype(np.float32)
        dummy_data.append(volume)
        # Random binary labels
        dummy_labels.append(bool(np.random.randint(0, 2)))

    df = pd.DataFrame({
        'data': dummy_data,
        'manual_classification': dummy_labels,
        'spot_id': range(n_spots)
    })

    print(f"Created {len(df)} spots ({sum(dummy_labels)} positive)")

    # Train model
    print("\nTraining SE-ResNet3D model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, history = train_model(
        df,
        learning_rate=1e-4,
        batch_size=16,
        epochs=50,
        test_size=0.3,
        device=device,
        use_augmentation=True,
        verbose=True
    )

    # Save model
    output_dir = Path("trained_models")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "se_resnet3d_basic.pt"

    print(f"\nSaving model to {model_path}")
    save_model(model, str(model_path), history=history)

    # Plot training curves
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(history['train_acc'], label='Train')
        axes[1].plot(history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Metrics
        axes[2].plot(history['val_precision'], label='Precision')
        axes[2].plot(history['val_recall'], label='Recall')
        axes[2].plot(history['val_auc'], label='AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Validation Metrics')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training curves to {plot_path}")

    except ImportError:
        print("matplotlib not available, skipping plot")

    return model, df


def example_kfold_training():
    """Example 2: K-Fold cross-validation training."""
    print("\n" + "="*60)
    print("Example 2: K-Fold Cross-Validation")
    print("="*60)

    # Create dummy data
    print("\nCreating dummy training data...")
    n_spots = 300
    dummy_data = []
    dummy_labels = []

    for i in range(n_spots):
        volume = np.random.rand(9, 11, 11).astype(np.float32)
        dummy_data.append(volume)
        dummy_labels.append(bool(np.random.randint(0, 2)))

    df = pd.DataFrame({
        'data': dummy_data,
        'manual_classification': dummy_labels,
        'spot_id': range(n_spots)
    })

    print(f"Created {len(df)} spots ({sum(dummy_labels)} positive)")

    # K-Fold training
    models_dir = Path("trained_models/kfold")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nTraining with 5-fold cross-validation on {device}...")
    train_model_kfold(
        df,
        models_dir=str(models_dir),
        n_splits=5,
        base_channels=32,
        learning_rate=1e-4,
        batch_size=16,
        epochs=30,
        device=device,
        verbose=True
    )

    print(f"\nModels saved to {models_dir}")
    print("Files created:")
    for f in sorted(models_dir.glob("*")):
        print(f"  - {f.name}")


def example_inference():
    """Example 3: Running inference on new data."""
    print("\n" + "="*60)
    print("Example 3: Inference on New Data")
    print("="*60)

    # First, ensure we have a trained model
    model_path = Path("trained_models/se_resnet3d_basic.pt")
    if not model_path.exists():
        print("\nNo trained model found. Training a quick model first...")
        model, _ = example_basic_training()
    else:
        print(f"\nLoading model from {model_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(str(model_path), device=device)

    # Create new spots to classify
    print("\nCreating test spots...")
    n_test = 50
    test_data = []
    for i in range(n_test):
        volume = np.random.rand(9, 11, 11).astype(np.float32)
        test_data.append(volume)

    test_df = pd.DataFrame({
        'data': test_data,
        'spot_id': range(n_test)
    })

    # Run prediction
    print("Running batch prediction...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result_df = run_batch_prediction(
        test_df,
        path_to_model=str(model_path),
        device=device,
        batch_size=32
    )

    print("\nPrediction results:")
    print(result_df[['spot_id', 'prob']].head(10))

    # Classify with threshold
    threshold = 0.5
    result_df['predicted_class'] = result_df['prob'] > threshold
    n_positive = result_df['predicted_class'].sum()

    print(f"\nClassification summary (threshold={threshold}):")
    print(f"  Positive: {n_positive} / {n_test} ({100*n_positive/n_test:.1f}%)")
    print(f"  Negative: {n_test - n_positive} / {n_test} ({100*(n_test-n_positive)/n_test:.1f}%)")

    return result_df


def example_model_comparison():
    """Example 4: Compare different model configurations."""
    print("\n" + "="*60)
    print("Example 4: Model Architecture Comparison")
    print("="*60)

    # Create test data
    n_spots = 150
    dummy_data = []
    dummy_labels = []

    for i in range(n_spots):
        volume = np.random.rand(9, 11, 11).astype(np.float32)
        dummy_data.append(volume)
        dummy_labels.append(bool(np.random.randint(0, 2)))

    df = pd.DataFrame({
        'data': dummy_data,
        'manual_classification': dummy_labels,
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = [
        {'name': 'Small (16 channels)', 'base_channels': 16, 'use_se': True},
        {'name': 'Medium (32 channels)', 'base_channels': 32, 'use_se': True},
        {'name': 'Large (64 channels)', 'base_channels': 64, 'use_se': True},
        {'name': 'Medium without SE', 'base_channels': 32, 'use_se': False},
    ]

    results = []

    for config in configs:
        print(f"\nTraining: {config['name']}")
        model = make_se_resnet3d(
            base_channels=config['base_channels'],
            use_se=config['use_se']
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Train briefly
        _, history = train_model(
            df, model=model,
            learning_rate=1e-4,
            batch_size=16,
            epochs=20,
            device=device,
            verbose=False
        )

        # Get final metrics
        final_acc = history['val_acc'][-1]
        final_auc = history['val_auc'][-1]

        results.append({
            'name': config['name'],
            'params': n_params,
            'val_acc': final_acc,
            'val_auc': final_auc
        })

        print(f"  Final Val Acc: {final_acc:.4f}")
        print(f"  Final Val AUC: {final_auc:.4f}")

    # Summary table
    print("\n" + "="*60)
    print("Comparison Summary:")
    print("="*60)
    print(f"{'Model':<25} {'Parameters':>12} {'Val Acc':>10} {'Val AUC':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<25} {r['params']:>12,} {r['val_acc']:>10.4f} {r['val_auc']:>10.4f}")


def main():
    """Run all examples."""
    print("\nPyTorch SE-ResNet3D Spot Classifier Examples")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU available, using CPU")

    # Run examples
    try:
        # Example 1: Basic training
        example_basic_training()

        # Example 2: K-Fold training
        example_kfold_training()

        # Example 3: Inference
        example_inference()

        # Example 4: Model comparison
        example_model_comparison()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
