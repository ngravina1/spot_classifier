"""Compare PyTorch Simple CNN vs PyTorch SE-ResNet3D models.

This script provides a clean interface for comparing both models on
the same training data using the same framework (PyTorch).

For cross-framework comparison (TensorFlow vs PyTorch), use --use-tensorflow flag.

Usage:
    python compare_models.py [--options]
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from zms2.spots.trainer import create_trainer
from zms2.spots.utils import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_summary,
    visualize_spot_examples
)


def load_training_data(data_path: str) -> tuple:
    """Load and prepare training data.

    Args:
        data_path: Path to training_data.pkl

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, df)
    """
    print(f"Loading data from: {data_path}")

    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    print(f"Loaded {len(df)} spots")

    # Extract data and labels
    data = np.array(df['data'].to_list())
    labels = np.array(df['manual_classification'].to_list(), dtype=int)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTraining: {len(X_train)} samples ({np.sum(y_train)} positive)")
    print(f"Validation: {len(X_val)} samples ({np.sum(y_val)} positive)")

    return X_train, X_val, y_train, y_val, df


def train_and_compare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    train_simple_cnn: bool = True,
    train_se_resnet: bool = True,
    use_tensorflow: bool = False
):
    """Train both models and compare results.

    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        output_dir: Directory to save outputs
        epochs: Number of training epochs
        batch_size: Training batch size
        train_simple_cnn: Whether to train Simple CNN
        train_se_resnet: Whether to train SE-ResNet
        use_tensorflow: If True, use TensorFlow for Simple CNN (old behavior)
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {}

    # Train Simple CNN (PyTorch or TensorFlow)
    if train_simple_cnn:
        if use_tensorflow:
            print("\n" + "="*70)
            print("TRAINING TENSORFLOW SIMPLE CNN")
            print("="*70)

            try:
                simple_trainer = create_trainer(
                    'tensorflow',
                    n_filters1=4,
                    n_filters2=4,
                    learning_rate=1e-4,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=True
                )

                simple_trainer.build_model()
                simple_history = simple_trainer.train(X_train, y_train, X_val, y_val)

                # Predictions
                simple_probs = simple_trainer.predict(X_val)
                simple_preds = (simple_probs > 0.5).astype(int)

                # Save model
                simple_trainer.save(output_dir / 'simple_cnn.h5')

                # Save results
                results['simple_cnn'] = {
                    'trainer': simple_trainer,
                    'history': simple_history,
                    'probs': simple_probs,
                    'preds': simple_preds
                }

                # Plot training curves
                plot_training_curves(
                    simple_history,
                    save_path=output_dir / 'simple_cnn_training.png',
                    show=False
                )

            except Exception as e:
                print(f"Error training TensorFlow model: {e}")
                print("Skipping TensorFlow model...")
                results['simple_cnn'] = None
        else:
            print("\n" + "="*70)
            print("TRAINING PYTORCH SIMPLE CNN")
            print("="*70)

            simple_trainer = create_trainer(
                'simple_cnn',
                n_filters1=4,
                n_filters2=4,
                learning_rate=1e-4,
                batch_size=batch_size,
                epochs=epochs,
                device='auto',
                verbose=True
            )

            simple_trainer.build_model()
            simple_history = simple_trainer.train(X_train, y_train, X_val, y_val)

            # Predictions
            simple_probs = simple_trainer.predict(X_val)
            simple_preds = (simple_probs > 0.5).astype(int)

            # Save model
            simple_trainer.save(output_dir / 'simple_cnn.pt')

            # Save results
            results['simple_cnn'] = {
                'trainer': simple_trainer,
                'history': simple_history,
                'probs': simple_probs,
                'preds': simple_preds
            }

            # Plot training curves
            plot_training_curves(
                simple_history,
                save_path=output_dir / 'simple_cnn_training.png',
                show=False
            )

    # Train SE-ResNet3D
    if train_se_resnet:
        print("\n" + "="*70)
        print("TRAINING PYTORCH SE-RESNET3D")
        print("="*70)

        resnet_trainer = create_trainer(
            'pytorch',
            base_channels=32,
            use_se=True,
            learning_rate=1e-4,
            batch_size=batch_size,
            epochs=epochs,
            device='auto',
            verbose=True
        )

        resnet_trainer.build_model()
        resnet_history = resnet_trainer.train(X_train, y_train, X_val, y_val)

        # Predictions
        resnet_probs = resnet_trainer.predict(X_val)
        resnet_preds = (resnet_probs > 0.5).astype(int)

        # Save model
        resnet_trainer.save(output_dir / 'se_resnet3d.pt')

        # Save results
        results['se_resnet'] = {
            'trainer': resnet_trainer,
            'history': resnet_history,
            'probs': resnet_probs,
            'preds': resnet_preds
        }

        # Plot training curves
        plot_training_curves(
            resnet_history,
            save_path=output_dir / 'se_resnet_training.png',
            show=False
        )

    # Comparison visualizations
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)

    # Confusion matrices
    if results.get('simple_cnn'):
        plot_confusion_matrix(
            y_val, results['simple_cnn']['preds'],
            title='Simple CNN - Confusion Matrix',
            save_path=output_dir / 'simple_cnn_cm.png',
            show=False
        )

    if results.get('se_resnet'):
        plot_confusion_matrix(
            y_val, results['se_resnet']['preds'],
            title='SE-ResNet3D - Confusion Matrix',
            save_path=output_dir / 'se_resnet_cm.png',
            show=False
        )

    # ROC curves
    if results.get('simple_cnn'):
        plot_roc_curve(
            y_val, results['simple_cnn']['probs'],
            title='Simple CNN - ROC Curve',
            save_path=output_dir / 'simple_cnn_roc.png',
            show=False
        )

    if results.get('se_resnet'):
        plot_roc_curve(
            y_val, results['se_resnet']['probs'],
            title='SE-ResNet3D - ROC Curve',
            save_path=output_dir / 'se_resnet_roc.png',
            show=False
        )

    # Classification reports
    print("\n" + "="*70)
    print("CLASSIFICATION REPORTS")
    print("="*70)

    if results.get('simple_cnn'):
        print("\nSimple CNN:")
        print_classification_summary(
            y_val,
            results['simple_cnn']['preds'],
            results['simple_cnn']['probs']
        )

    if results.get('se_resnet'):
        print("\nSE-ResNet3D:")
        print_classification_summary(
            y_val,
            results['se_resnet']['preds'],
            results['se_resnet']['probs']
        )

    # Save summary
    import torch
    summary_data = {'Metric': []}
    if results.get('simple_cnn'):
        summary_data['Simple CNN'] = []
    if results.get('se_resnet'):
        summary_data['SE-ResNet3D'] = []

    metrics = [
        ('Parameters', 'n_params'),
        ('Training Time (min)', 'train_time'),
        ('Val Accuracy', 'val_acc'),
        ('Val Precision', 'val_precision'),
        ('Val Recall', 'val_recall'),
        ('Val AUC', 'val_auc')
    ]

    for metric_name, metric_key in metrics:
        summary_data['Metric'].append(metric_name)

        if results.get('simple_cnn'):
            if metric_key == 'n_params':
                # Handle both TensorFlow and PyTorch models
                try:
                    val = results['simple_cnn']['trainer'].model.count_params()
                except AttributeError:
                    val = sum(p.numel() for p in results['simple_cnn']['trainer'].model.parameters())
                summary_data['Simple CNN'].append(f"{val:,}")
            elif metric_key == 'train_time':
                val = results['simple_cnn']['history']['train_time'] / 60
                summary_data['Simple CNN'].append(f"{val:.1f}")
            else:
                val = results['simple_cnn']['history'][metric_key][-1]
                summary_data['Simple CNN'].append(f"{val:.4f}")

        if results.get('se_resnet'):
            if metric_key == 'n_params':
                val = sum(p.numel() for p in results['se_resnet']['trainer'].model.parameters())
                summary_data['SE-ResNet3D'].append(f"{val:,}")
            elif metric_key == 'train_time':
                val = results['se_resnet']['history']['train_time'] / 60
                summary_data['SE-ResNet3D'].append(f"{val:.1f}")
            else:
                val = results['se_resnet']['history'][metric_key][-1]
                summary_data['SE-ResNet3D'].append(f"{val:.4f}")

    # Add improvement column if both models trained
    if results.get('simple_cnn') and results.get('se_resnet'):
        summary_data['Improvement'] = []

        simple_params = sum(p.numel() for p in results['simple_cnn']['trainer'].model.parameters()) if hasattr(results['simple_cnn']['trainer'].model.parameters, '__iter__') else results['simple_cnn']['trainer'].model.count_params()
        resnet_params = sum(p.numel() for p in results['se_resnet']['trainer'].model.parameters())

        summary_data['Improvement'].append(f"+{resnet_params - simple_params:,}")
        summary_data['Improvement'].append(f"{results['se_resnet']['history']['train_time']/results['simple_cnn']['history']['train_time']:.2f}x")

        for _, metric_key in metrics[2:]:
            simple_val = results['simple_cnn']['history'][metric_key][-1]
            resnet_val = results['se_resnet']['history'][metric_key][-1]
            diff = (resnet_val - simple_val) * 100
            summary_data['Improvement'].append(f"+{diff:.1f}%")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)

    # Visualize spot examples
    if results.get('se_resnet'):
        visualize_spot_examples(
            X_val,
            y_val,
            predictions=results['se_resnet']['preds'],
            n_examples=16,
            save_path=output_dir / 'spot_examples.png',
            show=False
        )

    print(f"\n✓ All results saved to: {output_dir}")

    return results


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare PyTorch Simple CNN vs PyTorch SE-ResNet3D'
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
        default='comparison_results',
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
        '--skip-simple-cnn',
        action='store_true',
        help='Skip Simple CNN training'
    )
    parser.add_argument(
        '--skip-se-resnet',
        action='store_true',
        help='Skip SE-ResNet3D training'
    )
    parser.add_argument(
        '--use-tensorflow',
        action='store_true',
        help='Use TensorFlow for Simple CNN (cross-framework comparison)'
    )

    args = parser.parse_args()

    # Setup
    print("="*70)
    if args.use_tensorflow:
        print("MODEL COMPARISON: TensorFlow CNN vs PyTorch SE-ResNet3D")
    else:
        print("MODEL COMPARISON: PyTorch Simple CNN vs PyTorch SE-ResNet3D")
    print("="*70)

    # Load data
    X_train, X_val, y_train, y_val, df = load_training_data(args.data_path)

    # Train and compare
    output_dir = Path(args.output_dir)
    results = train_and_compare(
        X_train, y_train, X_val, y_val,
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_simple_cnn=not args.skip_simple_cnn,
        train_se_resnet=not args.skip_se_resnet,
        use_tensorflow=args.use_tensorflow
    )

    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
