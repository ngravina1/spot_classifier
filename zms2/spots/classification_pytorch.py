"""PyTorch implementation of spot classification with 3D SE-ResNet

This module provides a modern PyTorch implementation of 3D spot classification,
replacing the original TensorFlow/Keras implementation with a more powerful
3D Squeeze-and-Excitation ResNet architecture.

Key improvements:
- 3D SE-ResNet architecture with residual connections and channel attention
- Better gradient flow through skip connections
- Channel attention mechanisms for feature refinement
- More flexible training with PyTorch's dynamic computation graph
- Improved metrics tracking with torchmetrics

Input data structure: (spots, z, y, x), with currently (z,y,x) = (9,11,11)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchmetrics
from scipy import ndimage as ndi
from sklearn.model_selection import KFold
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import random


# =============================================================================
# Data Normalization
# =============================================================================

def normalize_data(data: np.ndarray) -> np.ndarray:
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


# =============================================================================
# 3D SE-ResNet Architecture
# =============================================================================

class SqueezeExcitation3D(nn.Module):
    """3D Squeeze-and-Excitation block for channel attention.

    This module learns to emphasize informative feature channels and
    suppress less useful ones through a channel attention mechanism.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the bottleneck (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        # Squeeze: global spatial pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: learn channel weights
        y = self.excitation(y).view(b, c, 1, 1, 1)
        # Scale input by channel weights
        return x * y.expand_as(x)


class ResidualBlock3D(nn.Module):
    """3D Residual block with optional Squeeze-and-Excitation.

    Implements a standard residual block with two 3D convolutions,
    batch normalization, and a skip connection. Optionally includes
    SE attention for improved feature learning.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (default: 1)
        use_se: Whether to use Squeeze-and-Excitation (default: True)
        se_reduction: SE reduction ratio (default: 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        # SE block
        self.se = SqueezeExcitation3D(out_channels, se_reduction) if use_se else None

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.se is not None:
            out = self.se(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class SEResNet3D(nn.Module):
    """3D Squeeze-and-Excitation ResNet for spot classification.

    Architecture:
    - Initial 3D convolution
    - 3 residual blocks with SE attention and progressive pooling
    - Global average pooling
    - Fully connected classification head

    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of output classes (default: 1 for binary)
        base_channels: Number of channels in first layer (default: 32)
        use_se: Whether to use SE blocks (default: True)
        dropout: Dropout rate (default: 0.5)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        use_se: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)

        # Residual blocks with progressive channel expansion
        self.layer1 = ResidualBlock3D(base_channels, base_channels,
                                      stride=1, use_se=use_se)
        self.pool1 = nn.MaxPool3d(2)

        self.layer2 = ResidualBlock3D(base_channels, base_channels * 2,
                                      stride=1, use_se=use_se)
        self.pool2 = nn.MaxPool3d(2)

        self.layer3 = ResidualBlock3D(base_channels * 2, base_channels * 4,
                                      stride=1, use_se=use_se)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 4, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # Residual blocks with pooling
        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)

        # Classification head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def make_se_resnet3d(
    depth: int = 9,
    width: int = 11,
    height: int = 11,
    base_channels: int = 32,
    use_se: bool = True,
    dropout: float = 0.5
) -> SEResNet3D:
    """Factory function to create SE-ResNet3D model.

    Args:
        depth: Z dimension (default: 9)
        width: X dimension (default: 11)
        height: Y dimension (default: 11)
        base_channels: Base number of channels (default: 32)
        use_se: Use SE attention blocks (default: True)
        dropout: Dropout rate (default: 0.5)

    Returns:
        SEResNet3D model instance
    """
    return SEResNet3D(
        in_channels=1,
        num_classes=1,
        base_channels=base_channels,
        use_se=use_se,
        dropout=dropout
    )


# =============================================================================
# Dataset and Data Loading
# =============================================================================

class SpotDataset(Dataset):
    """PyTorch Dataset for 3D spot volumes.

    Args:
        data: Array of shape (n_spots, z, y, x)
        labels: Array of binary labels (n_spots,)
        transform: Optional transform to apply to data
        normalize: Whether to normalize data (default: True)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        transform=None,
        normalize: bool = True
    ):
        if normalize:
            self.data = normalize_data(data)
        else:
            self.data = data.astype(np.float32)

        self.labels = labels.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            volume = self.transform(volume)

        # Add channel dimension: (z, y, x) -> (1, z, y, x)
        volume = torch.from_numpy(volume).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return volume, label


class RandomRotation3D:
    """Random 3D rotation augmentation.

    Args:
        angles: List of rotation angles to randomly choose from
    """

    def __init__(self, angles: List[float] = None):
        if angles is None:
            self.angles = [-20, -10, -5, 5, 10, 20]
        else:
            self.angles = angles

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        angle = random.choice(self.angles)
        rotated = ndi.rotate(volume, angle, reshape=False, order=1)
        # Clip to valid range
        rotated = np.clip(rotated, 0, 1)
        return rotated.astype(np.float32)


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    df: pd.DataFrame,
    model: Optional[nn.Module] = None,
    learning_rate: float = 1e-4,
    batch_size: int = 8,
    epochs: int = 100,
    test_size: float = 0.33,
    device: str = 'cuda',
    use_augmentation: bool = True,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """Train SE-ResNet3D model on spot data.

    Args:
        df: DataFrame with 'data' and 'manual_classification' columns
        model: Model to train (creates new if None)
        learning_rate: Adam learning rate (default: 1e-4)
        batch_size: Batch size (default: 8)
        epochs: Number of training epochs (default: 100)
        test_size: Validation split fraction (default: 0.33)
        device: Device to train on ('cuda' or 'cpu')
        use_augmentation: Whether to apply rotation augmentation
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_model, history_dict)
    """
    # Filter out unclassified spots
    df_filtered = df[[label is not None for label in df.manual_classification]]

    # Extract data and labels
    data = np.array(df_filtered.data.to_list())
    labels = np.array(df_filtered.manual_classification.to_list(), dtype=int)

    # Split data
    n_train = int(len(data) * (1 - test_size))
    indices = np.random.permutation(len(data))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Create datasets
    train_transform = RandomRotation3D() if use_augmentation else None
    train_dataset = SpotDataset(data[train_idx], labels[train_idx],
                               transform=train_transform)
    val_dataset = SpotDataset(data[val_idx], labels[val_idx])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0, pin_memory=True)

    # Create model if not provided
    if model is None:
        model = make_se_resnet3d()

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Calculate class weights for imbalanced data
    neg, pos = np.bincount(labels)
    total = neg + pos
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics
    train_acc = torchmetrics.Accuracy(task='binary').to(device)
    val_acc = torchmetrics.Accuracy(task='binary').to(device)
    val_precision = torchmetrics.Precision(task='binary').to(device)
    val_recall = torchmetrics.Recall(task='binary').to(device)
    val_auc = torchmetrics.AUROC(task='binary').to(device)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_auc': []
    }

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_epoch = 0
        train_acc.reset()

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_acc.update(preds, batch_labels.int())

        train_loss_epoch /= len(train_loader)
        train_acc_epoch = train_acc.compute().item()

        # Validation phase
        model.eval()
        val_loss_epoch = 0
        val_acc.reset()
        val_precision.reset()
        val_recall.reset()
        val_auc.reset()

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_data).squeeze()
                loss = criterion(outputs, batch_labels)
                val_loss_epoch += loss.item()

                probs = torch.sigmoid(outputs)
                preds = probs > 0.5

                val_acc.update(preds, batch_labels.int())
                val_precision.update(preds, batch_labels.int())
                val_recall.update(preds, batch_labels.int())
                val_auc.update(probs, batch_labels.int())

        val_loss_epoch /= len(val_loader)
        val_acc_epoch = val_acc.compute().item()
        val_prec_epoch = val_precision.compute().item()
        val_rec_epoch = val_recall.compute().item()
        val_auc_epoch = val_auc.compute().item()

        # Store history
        history['train_loss'].append(train_loss_epoch)
        history['train_acc'].append(train_acc_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['val_acc'].append(val_acc_epoch)
        history['val_precision'].append(val_prec_epoch)
        history['val_recall'].append(val_rec_epoch)
        history['val_auc'].append(val_auc_epoch)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss_epoch:.4f}, Acc: {train_acc_epoch:.4f}")
            print(f"  Val   - Loss: {val_loss_epoch:.4f}, Acc: {val_acc_epoch:.4f}, "
                  f"Prec: {val_prec_epoch:.4f}, Rec: {val_rec_epoch:.4f}, "
                  f"AUC: {val_auc_epoch:.4f}")

    if verbose:
        n_true = np.sum(labels[val_idx])
        print(f"\nTraining complete!")
        print(f"Number of true spots in validation set: {n_true}")

    return model, history


def train_model_kfold(
    df: pd.DataFrame,
    models_dir: str,
    n_splits: int = 5,
    base_channels: int = 32,
    learning_rate: float = 1e-4,
    batch_size: int = 8,
    epochs: int = 100,
    device: str = 'cuda',
    verbose: bool = True
) -> None:
    """Train SE-ResNet3D using K-Fold cross-validation.

    Args:
        df: DataFrame with 'data' and 'manual_classification' columns
        models_dir: Directory to save trained models
        n_splits: Number of K-Fold splits (default: 5)
        base_channels: Base channels for model (default: 32)
        learning_rate: Adam learning rate (default: 1e-4)
        batch_size: Batch size (default: 8)
        epochs: Number of training epochs (default: 100)
        device: Device to train on ('cuda' or 'cpu')
        verbose: Whether to print progress
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Extract data and labels
    data = np.array(df.data.to_list()).astype(np.float32)
    labels = np.array(df.manual_classification.to_list(), dtype=int)

    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Fold {fold}/{n_splits}")
            print(f"{'='*60}")

        # Create datasets
        train_dataset = SpotDataset(
            data[train_idx], labels[train_idx],
            transform=RandomRotation3D()
        )
        val_dataset = SpotDataset(data[val_idx], labels[val_idx])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)

        # Create fresh model for this fold
        model = make_se_resnet3d(base_channels=base_channels).to(device)

        # Class weights
        neg, pos = np.bincount(labels)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Metrics
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_data).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                    outputs = model(batch_data).squeeze()
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == batch_labels).sum().item()
                    total += batch_labels.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = models_dir / f"model_fold_{fold}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, model_path)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Val Acc improved to {val_acc:.4f} - Model saved")

        # Save history
        history_path = models_dir / f"history_fold_{fold}.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        if verbose:
            print(f"Fold {fold} complete - Best Val Acc: {best_val_acc:.4f}")


# =============================================================================
# Inference Functions
# =============================================================================

def make_prediction(
    model: nn.Module,
    data: np.ndarray,
    device: str = 'cuda',
    batch_size: int = 32
) -> np.ndarray:
    """Use trained model to classify spots.

    Args:
        model: Trained PyTorch model
        data: Array of shape (n_spots, z, y, x)
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        Array of probabilities (n_spots,)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Normalize data
    data_norm = normalize_data(data)
    dataset = SpotDataset(data_norm, np.zeros(len(data)), normalize=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    results = []
    with torch.no_grad():
        for batch_data, _ in loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data).squeeze()
            probs = torch.sigmoid(outputs)
            results.append(probs.cpu().numpy())

    return np.concatenate(results)


def run_batch_prediction(
    df: pd.DataFrame,
    path_to_model: str,
    device: str = 'cuda',
    batch_size: int = 32
) -> pd.DataFrame:
    """Run classifier on every spot in a DataFrame.

    Args:
        df: DataFrame with 'data' column
        path_to_model: Path to saved .pt model file
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        DataFrame with added 'prob' column
    """
    # Load model
    checkpoint = torch.load(path_to_model, map_location=device)
    model = make_se_resnet3d()

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Extract data
    data = np.array(df.data.to_list()).astype(np.float32)

    # Predict
    results = make_prediction(model, data, device=device, batch_size=batch_size)

    # Update DataFrame
    df = df.copy()
    df['prob'] = results

    return df


def run_batch_prediction_by_time_point(
    df: pd.DataFrame,
    path_to_model: str,
    device: str = 'cuda',
    batch_size: int = 32
) -> pd.DataFrame:
    """Run classifier on spots in chunks by time point.

    Args:
        df: DataFrame with 'data' and 't' columns
        path_to_model: Path to saved .pt model file
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        DataFrame with added 'prob' column
    """
    # Load model
    checkpoint = torch.load(path_to_model, map_location=device)
    model = make_se_resnet3d()

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    time_points = np.unique(df.t)
    prob_df = pd.DataFrame(np.zeros(len(df)), columns=['prob'],
                          dtype='float32', index=df.index)

    for t in time_points:
        print(f'Running prediction for t = {t}')
        this_df = df[df.t == t]
        data = np.array(this_df.data.to_list()).astype('float32')

        # Predict
        results = make_prediction(model, data, device=device, batch_size=batch_size)
        prob_df.loc[this_df.index, 'prob'] = results

    df = pd.concat((df, prob_df), axis=1)
    return df


# =============================================================================
# Utility Functions
# =============================================================================

def create_training_data_from_spots_df(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
    manual_labels: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pixel data and labels from spots DataFrame.

    Args:
        df: DataFrame with 'data' and 'manual_classification' columns
        save_dir: Optional directory to save training data
        manual_labels: If True, only extract manually labeled data

    Returns:
        Tuple of (data_array, labels_array)
    """
    # Check for manual classifications
    if manual_labels and not any(df.manual_classification):
        raise ValueError('No spots have been manually classified.')

    data = np.array(df.data.to_list())
    labels = np.array(df.manual_classification.to_list())

    if manual_labels:
        mask = labels != None
        data = data[mask]
        labels = labels[mask]

    if save_dir is not None and manual_labels:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'training_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        with open(save_dir / 'training_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)

    return data, labels


def save_model(model: nn.Module, path: str, **kwargs):
    """Save PyTorch model with optional metadata.

    Args:
        model: Model to save
        path: Save path (.pt extension)
        **kwargs: Additional metadata to save
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    torch.save(save_dict, path)


def load_model(path: str, device: str = 'cuda') -> nn.Module:
    """Load PyTorch model from file.

    Args:
        path: Path to .pt model file
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(path, map_location=device)
    model = make_se_resnet3d()

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model
