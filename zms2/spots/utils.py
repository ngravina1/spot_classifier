"""Utility functions for spot classification.

This module provides common utilities used across different classification
implementations (TensorFlow and PyTorch).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def normalize_spot_data(data: np.ndarray) -> np.ndarray:
    """Normalize voxel intensities to (0,1) per spot.

    Args:
        data: Array of shape (n_spots, z, y, x)

    Returns:
        Normalized array of same shape
    """
    data_norm = np.zeros_like(data, dtype=np.float32)
    for spot_ind in range(len(data)):
        spot = data[spot_ind]
        min_val = np.min(spot)
        max_val = np.max(spot)
        if max_val > min_val:
            data_norm[spot_ind] = (spot - min_val) / (max_val - min_val)
        else:
            data_norm[spot_ind] = spot
    return data_norm


def plot_training_curves(
    history: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot training curves from history dictionary.

    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Validation metrics
    axes[1, 0].plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
    axes[1, 0].plot(epochs, history['val_recall'], 'orange', label='Recall', linewidth=2)
    axes[1, 0].plot(epochs, history['val_auc'], 'purple', label='AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Confusion Matrix',
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = 'ROC Curve',
    save_path: Optional[Path] = None,
    show: bool = True
) -> float:
    """Plot ROC curve and return AUC.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot

    Returns:
        AUC score
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return roc_auc


def calculate_class_weights(labels: np.ndarray) -> Tuple[float, float]:
    """Calculate class weights for imbalanced data.

    Args:
        labels: Binary labels array

    Returns:
        Tuple of (weight_for_0, weight_for_1)
    """
    neg, pos = np.bincount(labels.astype(int))
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    return weight_for_0, weight_for_1


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> None:
    """Print detailed classification summary.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report
    )

    print("=" * 60)
    print("Classification Summary")
    print("=" * 60)

    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_true, y_prob)
        print(f"AUC:       {auc_score:.4f}")

    print("\n" + "=" * 60)
    print("Detailed Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=['Negative', 'Positive']))
    print("=" * 60)


def visualize_spot_examples(
    data: np.ndarray,
    labels: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    n_examples: int = 8,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Visualize example spots with their labels.

    Args:
        data: Spot data array (n_spots, z, y, x)
        labels: True labels
        predictions: Optional predicted labels
        n_examples: Number of examples to show
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    n_examples = min(n_examples, len(data))
    indices = np.random.choice(len(data), n_examples, replace=False)

    n_cols = 4
    n_rows = (n_examples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, idx in enumerate(indices):
        spot = data[idx]
        # Show middle z-slice
        z_mid = spot.shape[0] // 2
        im = spot[z_mid, :, :]

        axes[i].imshow(im, cmap='hot', interpolation='nearest')

        title = f"True: {'Pos' if labels[idx] else 'Neg'}"
        if predictions is not None:
            pred = predictions[idx]
            correct = (pred == labels[idx])
            title += f"\nPred: {'Pos' if pred else 'Neg'}"
            title += " ✓" if correct else " ✗"

        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_examples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Spot Examples (Middle Z-Slice)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
