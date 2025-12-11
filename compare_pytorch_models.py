"""Compare PyTorch Simple CNN vs PyTorch SE-ResNet3D.

This script provides a fair comparison using both models in the same framework.

Usage:
    python compare_pytorch_models.py [--options]
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from zms2.spots.trainer import create_trainer
from zms2.spots.utils import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_summary,
)


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare PyTorch Simple CNN vs SE-ResNet3D'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='zms2_trainingData/Zebrafish_MS2_spot_classification/training_data.pkl',
        help='Path to training data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='pytorch_comparison',
        help='Directory to save results'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )

    args = parser.parse_args()

    # Setup
    print("="*70)
    print("PyTorch Model Comparison: Simple CNN vs SE-ResNet3D")
    print("="*70)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        df = pickle.load(f)

    print(f"Loaded {len(df)} spots")

    # Extract and split data
    data = np.array(df['data'].to_list())
    labels = np.array(df['manual_classification'].to_list(), dtype=int)

    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTraining: {len(X_train)} samples ({np.sum(y_train)} positive)")
    print(f"Validation: {len(X_val)} samples ({np.sum(y_val)} positive)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {}

    # Train Simple CNN
    print("\n" + "="*70)
    print("TRAINING PYTORCH SIMPLE CNN (Original Architecture)")
    print("="*70)

    simple_trainer = create_trainer(
        'simple_cnn',
        n_filters1=4,
        n_filters2=4,
        learning_rate=1e-4,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        verbose=True
    )

    simple_trainer.build_model()
    simple_history = simple_trainer.train(X_train, y_train, X_val, y_val)

    # Predictions
    simple_probs = simple_trainer.predict(X_val)
    simple_preds = (simple_probs > 0.5).astype(int)

    # Save
    simple_trainer.save(output_dir / 'simple_cnn.pt')

    results['simple'] = {
        'trainer': simple_trainer,
        'history': simple_history,
        'probs': simple_probs,
        'preds': simple_preds
    }

    # Plot
    plot_training_curves(
        simple_history,
        save_path=output_dir / 'simple_cnn_training.png',
        show=False
    )

    # Train SE-ResNet
    print("\n" + "="*70)
    print("TRAINING PYTORCH SE-RESNET3D (New Architecture)")
    print("="*70)

    resnet_trainer = create_trainer(
        'pytorch',
        base_channels=32,
        use_se=True,
        learning_rate=1e-4,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        verbose=True
    )

    resnet_trainer.build_model()
    resnet_history = resnet_trainer.train(X_train, y_train, X_val, y_val)

    # Predictions
    resnet_probs = resnet_trainer.predict(X_val)
    resnet_preds = (resnet_probs > 0.5).astype(int)

    # Save
    resnet_trainer.save(output_dir / 'se_resnet3d.pt')

    results['resnet'] = {
        'trainer': resnet_trainer,
        'history': resnet_history,
        'probs': resnet_probs,
        'preds': resnet_preds
    }

    # Plot
    plot_training_curves(
        resnet_history,
        save_path=output_dir / 'se_resnet_training.png',
        show=False
    )

    # Comparison visualizations
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)

    # Side-by-side training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, args.epochs + 1)

    # Training Loss
    axes[0, 0].plot(epochs, simple_history['train_loss'], 'b-', label='Simple CNN', linewidth=2, alpha=0.7)
    axes[0, 0].plot(epochs, resnet_history['train_loss'], 'r-', label='SE-ResNet', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation Loss
    axes[0, 1].plot(epochs, simple_history['val_loss'], 'b-', label='Simple CNN', linewidth=2, alpha=0.7)
    axes[0, 1].plot(epochs, resnet_history['val_loss'], 'r-', label='SE-ResNet', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Validation Accuracy
    axes[1, 0].plot(epochs, simple_history['val_acc'], 'b-', label='Simple CNN', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, resnet_history['val_acc'], 'r-', label='SE-ResNet', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Accuracy', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.7, 1.0])

    # Validation AUC
    axes[1, 1].plot(epochs, simple_history['val_auc'], 'b-', label='Simple CNN', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, resnet_history['val_auc'], 'r-', label='SE-ResNet', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Validation AUC', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.7, 1.0])

    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Metrics comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    simple_final = [
        simple_history['val_acc'][-1],
        simple_history['val_precision'][-1],
        simple_history['val_recall'][-1],
        simple_history['val_auc'][-1]
    ]
    resnet_final = [
        resnet_history['val_acc'][-1],
        resnet_history['val_precision'][-1],
        resnet_history['val_recall'][-1],
        resnet_history['val_auc'][-1]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, simple_final, width, label='Simple CNN', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, resnet_final, width, label='SE-ResNet3D', color='coral', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Final Validation Metrics Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrices
    plot_confusion_matrix(
        y_val, simple_preds,
        title='Simple CNN - Confusion Matrix',
        save_path=output_dir / 'simple_cnn_cm.png',
        show=False
    )

    plot_confusion_matrix(
        y_val, resnet_preds,
        title='SE-ResNet3D - Confusion Matrix',
        save_path=output_dir / 'se_resnet_cm.png',
        show=False
    )

    # ROC curves
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    fpr_simple, tpr_simple, _ = roc_curve(y_val, simple_probs)
    fpr_resnet, tpr_resnet, _ = roc_curve(y_val, resnet_probs)

    auc_simple = auc(fpr_simple, tpr_simple)
    auc_resnet = auc(fpr_resnet, tpr_resnet)

    ax.plot(fpr_simple, tpr_simple, 'b-', linewidth=2,
            label=f'Simple CNN (AUC = {auc_simple:.3f})', alpha=0.7)
    ax.plot(fpr_resnet, tpr_resnet, 'r-', linewidth=2,
            label=f'SE-ResNet (AUC = {auc_resnet:.3f})', alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Classification reports
    print("\n" + "="*70)
    print("CLASSIFICATION REPORTS")
    print("="*70)

    print("\nSimple CNN:")
    print_classification_summary(y_val, simple_preds, simple_probs)

    print("\nSE-ResNet3D:")
    print_classification_summary(y_val, resnet_preds, resnet_probs)

    # Summary table
    import torch

    summary_data = {
        'Metric': [
            'Parameters',
            'Training Time (min)',
            'Val Accuracy',
            'Val Precision',
            'Val Recall',
            'Val AUC'
        ],
        'Simple CNN': [
            f"{sum(p.numel() for p in simple_trainer.model.parameters()):,}",
            f"{simple_history['train_time']/60:.1f}",
            f"{simple_history['val_acc'][-1]:.4f}",
            f"{simple_history['val_precision'][-1]:.4f}",
            f"{simple_history['val_recall'][-1]:.4f}",
            f"{simple_history['val_auc'][-1]:.4f}"
        ],
        'SE-ResNet3D': [
            f"{sum(p.numel() for p in resnet_trainer.model.parameters()):,}",
            f"{resnet_history['train_time']/60:.1f}",
            f"{resnet_history['val_acc'][-1]:.4f}",
            f"{resnet_history['val_precision'][-1]:.4f}",
            f"{resnet_history['val_recall'][-1]:.4f}",
            f"{resnet_history['val_auc'][-1]:.4f}"
        ]
    }

    # Calculate improvements
    acc_diff = resnet_history['val_acc'][-1] - simple_history['val_acc'][-1]
    auc_diff = resnet_history['val_auc'][-1] - simple_history['val_auc'][-1]

    summary_data['Improvement'] = [
        f"+{sum(p.numel() for p in resnet_trainer.model.parameters()) - sum(p.numel() for p in simple_trainer.model.parameters()):,}",
        f"{resnet_history['train_time']/simple_history['train_time']:.2f}x",
        f"+{acc_diff*100:.1f}%",
        f"+{(resnet_history['val_precision'][-1] - simple_history['val_precision'][-1])*100:.1f}%",
        f"+{(resnet_history['val_recall'][-1] - simple_history['val_recall'][-1])*100:.1f}%",
        f"+{auc_diff*100:.1f}%"
    ]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)

    print(f"\n✓ All results saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")

    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
