"""Unified training interface for spot classification models.

This module provides a clean, consistent API for training both TensorFlow
and PyTorch models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from abc import ABC, abstractmethod
import time


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""

    def __init__(
        self,
        model_type: str,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 100,
        device: str = 'auto',
        verbose: bool = True
    ):
        """Initialize trainer.

        Args:
            model_type: Type of model ('tensorflow' or 'pytorch')
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Number of training epochs
            device: Device to train on ('cuda', 'cpu', or 'auto')
            verbose: Whether to print training progress
        """
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.model = None
        self.history = None

    @abstractmethod
    def build_model(self, **kwargs):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]):
        """Save the model."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]):
        """Load a saved model."""
        pass


class SimpleCNNTrainer(BaseTrainer):
    """Trainer for simple PyTorch 3D CNN (original architecture)."""

    def __init__(self, n_filters1: int = 4, n_filters2: int = 4, **kwargs):
        """Initialize simple CNN trainer.

        Args:
            n_filters1: Number of filters in first conv layer
            n_filters2: Number of filters in second conv layer
            **kwargs: Additional arguments passed to BaseTrainer
        """
        super().__init__(model_type='pytorch_simple_cnn', **kwargs)
        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2

    def build_model(self, **kwargs):
        """Build simple 3D CNN model."""
        from zms2.spots.classification_pytorch import make_simple_cnn3d
        import torch

        # Handle device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = make_simple_cnn3d(
            n_filters1=self.n_filters1,
            n_filters2=self.n_filters2,
            **kwargs
        )
        self.model = self.model.to(torch.device(self.device))

        if self.verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Built PyTorch Simple 3D CNN:")
            print(f"  Parameters: {n_params:,}")
            print(f"  Device: {self.device}")

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train the simple CNN model."""
        import torch
        import torch.nn as nn
        import torchmetrics
        from torch.utils.data import DataLoader
        from zms2.spots.classification_pytorch import SpotDataset, RandomRotation3D

        if self.model is None:
            self.build_model()

        device = torch.device(self.device)

        # Create datasets
        train_dataset = SpotDataset(X_train, y_train, transform=RandomRotation3D())
        val_dataset = SpotDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                   shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=0)

        # Loss and optimizer
        neg, pos = np.bincount(y_train.astype(int))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Metrics
        metrics = {
            'train_acc': torchmetrics.Accuracy(task='binary').to(device),
            'val_acc': torchmetrics.Accuracy(task='binary').to(device),
            'val_prec': torchmetrics.Precision(task='binary').to(device),
            'val_rec': torchmetrics.Recall(task='binary').to(device),
            'val_auc': torchmetrics.AUROC(task='binary').to(device)
        }

        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_auc': []
        }

        if self.verbose:
            print(f"\nTraining PyTorch Simple 3D CNN:")
            print(f"  Epochs: {self.epochs}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Class weight: {pos_weight.item():.3f}")
            print()

        start_time = time.time()

        # Training loop (same as PyTorchTrainer)
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            metrics['train_acc'].reset()

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(batch_data).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                metrics['train_acc'].update(preds, batch_labels.int())

            train_loss /= len(train_loader)
            train_acc = metrics['train_acc'].compute().item()

            # Validation
            self.model.eval()
            val_loss = 0
            for key in ['val_acc', 'val_prec', 'val_rec', 'val_auc']:
                metrics[key].reset()

            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = self.model(batch_data).squeeze()
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5

                    metrics['val_acc'].update(preds, batch_labels.int())
                    metrics['val_prec'].update(preds, batch_labels.int())
                    metrics['val_rec'].update(preds, batch_labels.int())
                    metrics['val_auc'].update(probs, batch_labels.int())

            val_loss /= len(val_loader)
            val_acc = metrics['val_acc'].compute().item()
            val_prec = metrics['val_prec'].compute().item()
            val_rec = metrics['val_rec'].compute().item()
            val_auc = metrics['val_auc'].compute().item()

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_prec)
            self.history['val_recall'].append(val_rec)
            self.history['val_auc'].append(val_auc)

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Loss: {train_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"Val AUC: {val_auc:.4f}")

        train_time = time.time() - start_time

        if self.verbose:
            print(f"\nTraining complete in {train_time/60:.1f} minutes")
            print(f"  Final val accuracy: {self.history['val_acc'][-1]:.4f}")
            print(f"  Final val AUC: {self.history['val_auc'][-1]:.4f}")

        self.history['train_time'] = train_time
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on data."""
        import torch
        from torch.utils.data import DataLoader
        from zms2.spots.classification_pytorch import SpotDataset

        if self.model is None:
            raise ValueError("Model not trained or loaded")

        device = torch.device(self.device)
        self.model.eval()

        dataset = SpotDataset(X, np.zeros(len(X)), normalize=True)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=0)

        probs = []
        with torch.no_grad():
            for batch_data, _ in loader:
                batch_data = batch_data.to(device)
                outputs = self.model(batch_data).squeeze()
                batch_probs = torch.sigmoid(outputs)
                probs.append(batch_probs.cpu().numpy())

        return np.concatenate(probs)

    def save(self, path: Union[str, Path]):
        """Save the model."""
        import torch

        path = Path(path)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': {
                'n_filters1': self.n_filters1,
                'n_filters2': self.n_filters2,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }
        torch.save(save_dict, path)

        if self.verbose:
            print(f"Saved model to: {path}")

    def load(self, path: Union[str, Path]):
        """Load a saved model."""
        import torch

        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        # Load config
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.n_filters1 = config.get('n_filters1', 4)
            self.n_filters2 = config.get('n_filters2', 4)

        # Build and load model
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if self.verbose:
            print(f"Loaded model from: {path}")


class PyTorchTrainer(BaseTrainer):
    """Trainer for PyTorch SE-ResNet3D models."""

    def __init__(self, base_channels: int = 32, use_se: bool = True, **kwargs):
        """Initialize PyTorch trainer.

        Args:
            base_channels: Number of base channels for SE-ResNet
            use_se: Whether to use Squeeze-and-Excitation blocks
            **kwargs: Additional arguments passed to BaseTrainer
        """
        super().__init__(model_type='pytorch', **kwargs)
        self.base_channels = base_channels
        self.use_se = use_se

    def build_model(self, **kwargs):
        """Build SE-ResNet3D model."""
        from zms2.spots.classification_pytorch import make_se_resnet3d
        import torch

        # Handle device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = make_se_resnet3d(
            base_channels=self.base_channels,
            use_se=self.use_se,
            **kwargs
        )
        self.model = self.model.to(torch.device(self.device))

        if self.verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Built PyTorch SE-ResNet3D:")
            print(f"  Parameters: {n_params:,}")
            print(f"  Device: {self.device}")

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train the PyTorch model."""
        import torch
        import torch.nn as nn
        import torchmetrics
        from torch.utils.data import DataLoader
        from zms2.spots.classification_pytorch import SpotDataset, RandomRotation3D

        if self.model is None:
            self.build_model()

        device = torch.device(self.device)

        # Create datasets
        train_dataset = SpotDataset(X_train, y_train, transform=RandomRotation3D())
        val_dataset = SpotDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                   shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=0)

        # Loss and optimizer
        neg, pos = np.bincount(y_train.astype(int))
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Metrics
        metrics = {
            'train_acc': torchmetrics.Accuracy(task='binary').to(device),
            'val_acc': torchmetrics.Accuracy(task='binary').to(device),
            'val_prec': torchmetrics.Precision(task='binary').to(device),
            'val_rec': torchmetrics.Recall(task='binary').to(device),
            'val_auc': torchmetrics.AUROC(task='binary').to(device)
        }

        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_auc': []
        }

        if self.verbose:
            print(f"\nTraining PyTorch SE-ResNet3D:")
            print(f"  Epochs: {self.epochs}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Class weight: {pos_weight.item():.3f}")
            print()

        start_time = time.time()

        # Training loop
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            metrics['train_acc'].reset()

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(batch_data).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                metrics['train_acc'].update(preds, batch_labels.int())

            train_loss /= len(train_loader)
            train_acc = metrics['train_acc'].compute().item()

            # Validation
            self.model.eval()
            val_loss = 0
            for key in ['val_acc', 'val_prec', 'val_rec', 'val_auc']:
                metrics[key].reset()

            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = self.model(batch_data).squeeze()
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5

                    metrics['val_acc'].update(preds, batch_labels.int())
                    metrics['val_prec'].update(preds, batch_labels.int())
                    metrics['val_rec'].update(preds, batch_labels.int())
                    metrics['val_auc'].update(probs, batch_labels.int())

            val_loss /= len(val_loader)
            val_acc = metrics['val_acc'].compute().item()
            val_prec = metrics['val_prec'].compute().item()
            val_rec = metrics['val_rec'].compute().item()
            val_auc = metrics['val_auc'].compute().item()

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_prec)
            self.history['val_recall'].append(val_rec)
            self.history['val_auc'].append(val_auc)

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Loss: {train_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"Val AUC: {val_auc:.4f}")

        train_time = time.time() - start_time

        if self.verbose:
            print(f"\nTraining complete in {train_time/60:.1f} minutes")
            print(f"  Final val accuracy: {self.history['val_acc'][-1]:.4f}")
            print(f"  Final val AUC: {self.history['val_auc'][-1]:.4f}")

        self.history['train_time'] = train_time
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on data."""
        import torch
        from torch.utils.data import DataLoader
        from zms2.spots.classification_pytorch import SpotDataset

        if self.model is None:
            raise ValueError("Model not trained or loaded")

        device = torch.device(self.device)
        self.model.eval()

        dataset = SpotDataset(X, np.zeros(len(X)), normalize=True)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=0)

        probs = []
        with torch.no_grad():
            for batch_data, _ in loader:
                batch_data = batch_data.to(device)
                outputs = self.model(batch_data).squeeze()
                batch_probs = torch.sigmoid(outputs)
                probs.append(batch_probs.cpu().numpy())

        return np.concatenate(probs)

    def save(self, path: Union[str, Path]):
        """Save the model."""
        import torch

        path = Path(path)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': {
                'base_channels': self.base_channels,
                'use_se': self.use_se,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }
        torch.save(save_dict, path)

        if self.verbose:
            print(f"Saved model to: {path}")

    def load(self, path: Union[str, Path]):
        """Load a saved model."""
        import torch

        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        # Load config
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.base_channels = config.get('base_channels', 32)
            self.use_se = config.get('use_se', True)

        # Build and load model
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if self.verbose:
            print(f"Loaded model from: {path}")


class TensorFlowTrainer(BaseTrainer):
    """Trainer for TensorFlow/Keras CNN models."""

    def __init__(self, n_filters1: int = 4, n_filters2: int = 4, **kwargs):
        """Initialize TensorFlow trainer.

        Args:
            n_filters1: Number of filters in first conv layer
            n_filters2: Number of filters in second conv layer
            **kwargs: Additional arguments passed to BaseTrainer
        """
        super().__init__(model_type='tensorflow', **kwargs)
        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2

    def build_model(self, **kwargs):
        """Build Keras 3D CNN model."""
        from zms2.spots.classification import make_cnn
        import tensorflow as tf

        self.model = make_cnn(
            width=11,
            height=11,
            depth=9,
            n_filters1=self.n_filters1,
            n_filters2=self.n_filters2,
            **kwargs
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )

        if self.verbose:
            n_params = self.model.count_params()
            print(f"Built TensorFlow 3D CNN:")
            print(f"  Parameters: {n_params:,}")

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train the TensorFlow model."""
        from zms2.spots.classification import normalize_data
        from zms2.spots.utils import calculate_class_weights

        if self.model is None:
            self.build_model()

        # Normalize data
        X_train_norm = normalize_data(X_train)
        X_val_norm = normalize_data(X_val)

        # Reshape (add channel dimension)
        X_train_tf = np.expand_dims(X_train_norm, axis=-1)
        X_val_tf = np.expand_dims(X_val_norm, axis=-1)

        # Class weights
        weight_0, weight_1 = calculate_class_weights(y_train)
        class_weight = {0: weight_0, 1: weight_1}

        if self.verbose:
            print(f"\nTraining TensorFlow 3D CNN:")
            print(f"  Epochs: {self.epochs}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Class weights: {{0: {weight_0:.3f}, 1: {weight_1:.3f}}}")
            print()

        start_time = time.time()

        # Train
        history = self.model.fit(
            X_train_tf, y_train,
            validation_data=(X_val_tf, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            verbose=1 if self.verbose else 0
        )

        train_time = time.time() - start_time

        # Convert history to standard format
        self.history = {
            'train_loss': history.history['loss'],
            'train_acc': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_acc': history.history['val_accuracy'],
            'val_precision': history.history['val_precision'],
            'val_recall': history.history['val_recall'],
            'val_auc': history.history['val_auc'],
            'train_time': train_time
        }

        if self.verbose:
            print(f"\nTraining complete in {train_time/60:.1f} minutes")
            print(f"  Final val accuracy: {self.history['val_acc'][-1]:.4f}")
            print(f"  Final val AUC: {self.history['val_auc'][-1]:.4f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on data."""
        from zms2.spots.classification import normalize_data

        if self.model is None:
            raise ValueError("Model not trained or loaded")

        X_norm = normalize_data(X)
        X_tf = np.expand_dims(X_norm, axis=-1)

        probs = self.model.predict(X_tf, batch_size=self.batch_size, verbose=0)
        return probs.flatten()

    def save(self, path: Union[str, Path]):
        """Save the model."""
        import pickle

        path = Path(path)

        # Save model
        self.model.save(str(path))

        # Save history separately
        if self.history:
            history_path = path.parent / (path.stem + '_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.history, f)

        if self.verbose:
            print(f"Saved model to: {path}")

    def load(self, path: Union[str, Path]):
        """Load a saved model."""
        import tensorflow as tf
        import pickle

        path = Path(path)
        self.model = tf.keras.models.load_model(str(path))

        # Try to load history
        history_path = path.parent / (path.stem + '_history.pkl')
        if history_path.exists():
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)

        if self.verbose:
            print(f"Loaded model from: {path}")


def create_trainer(
    model_type: str = 'pytorch',
    **kwargs
) -> BaseTrainer:
    """Factory function to create a trainer.

    Args:
        model_type: Type of model:
            - 'pytorch' or 'se_resnet': PyTorch SE-ResNet3D (new, advanced)
            - 'simple_cnn' or 'pytorch_cnn': PyTorch Simple CNN (original architecture)
            - 'tensorflow' or 'keras' or 'tf': TensorFlow/Keras CNN
        **kwargs: Arguments passed to trainer

    Returns:
        Trainer instance
    """
    model_type_lower = model_type.lower()

    if model_type_lower in ['pytorch', 'se_resnet', 'pytorch_se_resnet']:
        return PyTorchTrainer(**kwargs)
    elif model_type_lower in ['simple_cnn', 'pytorch_cnn', 'pytorch_simple_cnn', 'cnn']:
        return SimpleCNNTrainer(**kwargs)
    elif model_type_lower in ['tensorflow', 'keras', 'tf']:
        return TensorFlowTrainer(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: 'pytorch', 'simple_cnn', or 'tensorflow'"
        )
