"""Train PyTorch SE-ResNet3D using existing zms2 training data.

This script loads the pre-labeled training data from zms2_trainingData/
and trains the new PyTorch SE-ResNet3D model.

Usage:
    python train_with_existing_data.py [--options]
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Import PyTorch classification functions
from zms2.spots.classification_pytorch import (
    make_se_resnet3d,
    train_model,
    train_model_kfold,
    save_model,
)


def load_training_data(data_path: str) -> pd.DataFrame:
    """Load training data from pickle file.

    Args:
        data_path: Path to training_data.pkl file

    Returns:
        DataFrame with 'data' and 'manual_classification' columns
    """
    print(f"Loading training data from: {data_path}")

    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    print(f"Loaded {len(df)} spots")
    print(f"\nColumns: {df.columns.tolist()}")

    # Check for required columns
    if 'data' not in df.columns:
        raise ValueError("Training data must have 'data' column with 3D arrays")
    if 'manual_classification' not in df.columns:
        raise ValueError("Training data must have 'manual_classification' column")

    # Verify data shapes
    first_shape = df['data'].iloc[0].shape
    print(f"Spot dimensions: {first_shape}")

    # Check class distribution
    label_counts = df['manual_classification'].value_counts()
    print(f"\nClass distribution:")
    print(f"  True (positive):  {label_counts.get(True, 0)} ({100*label_counts.get(True, 0)/len(df):.1f}%)")
    print(f"  False (negative): {label_counts.get(False, 0)} ({100*label_counts.get(False, 0)/len(df):.1f}%)")

    return df


def train_basic_model(
    df: pd.DataFrame,
    output_dir: Path,
    base_channels: int = 32,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 100,
    test_size: float = 0.2,
    device: str = 'cuda'
):
    """Train a single model with train/validation split.

    Args:
        df: Training DataFrame
        output_dir: Directory to save outputs
        base_channels: Number of base channels for SE-ResNet
        learning_rate: Learning rate for Adam optimizer
        batch_size: Training batch size
        epochs: Number of training epochs
        test_size: Fraction for validation set
        device: Device to train on ('cuda' or 'cpu')
    """
    print("\n" + "="*70)
    print("TRAINING SINGLE MODEL")
    print("="*70)

    # Create model
    print(f"\nCreating SE-ResNet3D with {base_channels} base channels...")
    model = make_se_resnet3d(base_channels=base_channels, use_se=True)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    print(f"\nTraining on {device}...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Validation split: {test_size:.0%}")

    model, history = train_model(
        df,
        model=model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        test_size=test_size,
        device=device,
        use_augmentation=True,
        verbose=True
    )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"se_resnet3d_{timestamp}.pt"
    print(f"\nSaving model to: {model_path}")
    save_model(model, str(model_path), history=history, config={
        'base_channels': base_channels,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'test_size': test_size,
    })

    # Plot training curves
    plot_training_history(history, output_dir / f"training_curves_{timestamp}.png")

    # Save history
    history_path = output_dir / f"training_history_{timestamp}.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"\nTraining complete!")
    print(f"  Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  Final validation AUC: {history['val_auc'][-1]:.4f}")

    return model, history


def train_kfold_models(
    df: pd.DataFrame,
    output_dir: Path,
    n_splits: int = 5,
    base_channels: int = 32,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 100,
    device: str = 'cuda'
):
    """Train models using K-Fold cross-validation.

    Args:
        df: Training DataFrame
        output_dir: Directory to save outputs
        n_splits: Number of folds
        base_channels: Number of base channels for SE-ResNet
        learning_rate: Learning rate for Adam optimizer
        batch_size: Training batch size
        epochs: Number of training epochs per fold
        device: Device to train on ('cuda' or 'cpu')
    """
    print("\n" + "="*70)
    print(f"TRAINING {n_splits}-FOLD CROSS-VALIDATION")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kfold_dir = output_dir / f"kfold_{n_splits}fold_{timestamp}"
    kfold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {kfold_dir}")
    print(f"  Base channels: {base_channels}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per fold: {epochs}")
    print(f"  Device: {device}")

    train_model_kfold(
        df,
        models_dir=str(kfold_dir),
        n_splits=n_splits,
        base_channels=base_channels,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        verbose=True
    )

    print(f"\nK-Fold training complete!")
    print(f"Models saved in: {kfold_dir}")

    # Load and summarize results
    summarize_kfold_results(kfold_dir, n_splits)


def plot_training_history(history: dict, save_path: Path):
    """Plot training curves.

    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Validation metrics
    axes[1, 0].plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
    axes[1, 0].plot(epochs, history['val_recall'], 'orange', label='Recall', linewidth=2)
    axes[1, 0].plot(epochs, history['val_auc'], 'purple', label='AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ================

    Final Metrics:
    • Val Accuracy:  {history['val_acc'][-1]:.4f}
    • Val Precision: {history['val_precision'][-1]:.4f}
    • Val Recall:    {history['val_recall'][-1]:.4f}
    • Val AUC:       {history['val_auc'][-1]:.4f}

    Best Metrics:
    • Best Accuracy: {max(history['val_acc']):.4f} (epoch {np.argmax(history['val_acc'])+1})
    • Best AUC:      {max(history['val_auc']):.4f} (epoch {np.argmax(history['val_auc'])+1})

    Total Epochs: {len(epochs)}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def summarize_kfold_results(kfold_dir: Path, n_splits: int):
    """Summarize K-Fold cross-validation results.

    Args:
        kfold_dir: Directory containing K-Fold results
        n_splits: Number of folds
    """
    print("\n" + "="*70)
    print("K-FOLD RESULTS SUMMARY")
    print("="*70)

    val_accs = []

    for fold in range(1, n_splits + 1):
        history_path = kfold_dir / f"history_fold_{fold}.pkl"
        if history_path.exists():
            with open(history_path, 'rb') as f:
                history = pickle.load(f)

            best_acc = max(history['val_acc'])
            val_accs.append(best_acc)
            print(f"Fold {fold}: Best Val Accuracy = {best_acc:.4f}")

    if val_accs:
        print(f"\nOverall Statistics:")
        print(f"  Mean Val Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"  Min Val Accuracy:  {np.min(val_accs):.4f}")
        print(f"  Max Val Accuracy:  {np.max(val_accs):.4f}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train SE-ResNet3D on existing zms2 training data'
    )

    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default='zms2_trainingData/Zebrafish_MS2_spot_classification/training_data.pkl',
        help='Path to training_data.pkl file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_models',
        help='Directory to save trained models'
    )

    # Training mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'kfold', 'both'],
        default='single',
        help='Training mode: single model, k-fold, or both'
    )

    # Model arguments
    parser.add_argument(
        '--base-channels',
        type=int,
        default=32,
        help='Number of base channels for SE-ResNet (default: 32)'
    )

    # Training arguments
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs (default: 100)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Validation split fraction for single mode (default: 0.2)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of folds for k-fold mode (default: 5)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to train on (default: auto)'
    )

    args = parser.parse_args()

    # Setup
    print("="*70)
    print("PyTorch SE-ResNet3D Training")
    print("="*70)

    # Check device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("\nWARNING: CUDA requested but not available. Using CPU.")
            device = 'cpu'
    else:
        print(f"\nUsing CPU (training will be slower)")

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at: {data_path}")

    df = load_training_data(str(data_path))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Train
    if args.mode in ['single', 'both']:
        train_basic_model(
            df,
            output_dir,
            base_channels=args.base_channels,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            test_size=args.test_size,
            device=device
        )

    if args.mode in ['kfold', 'both']:
        train_kfold_models(
            df,
            output_dir,
            n_splits=args.n_splits,
            base_channels=args.base_channels,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device
        )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
