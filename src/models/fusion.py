import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import numpy as np

from .encoders import MRIEncoder, ClinicalEncoder, TimeLapsedEncoder


class FusionRegressor(pl.LightningModule):
    def __init__(self, mri_dim=2048, clinical_dim=3, time_dim=256, hidden_dims=[256, 512, 1024, 512, 256]):
        """
        Model that combines information from MRI, clinical data, and time to predict current and future scores
        
        Args:
            mri_dim (int): Dimension of MRI feature vector
            clinical_dim (int): Dimension of clinical data vector
            time_dim (int): Dimension of time feature vector
            hidden_dims (list): List of hidden layer dimensions
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders
        self.mri_encoder = MRIEncoder(feature_dim=mri_dim)
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_dim)
        self.time_encoder = TimeLapsedEncoder(output_dim=time_dim)
        
        # Current scores prediction (without time)
        current_input_dim = mri_dim + self.clinical_encoder.get_feature_dim()
        
        # Shared layers for current scores
        current_layers = []
        prev_dim = current_input_dim
        for hidden_dim in hidden_dims:
            current_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.current_shared = nn.Sequential(*current_layers)
        
        # Separate branches for current scores
        self.current_adas11 = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prev_dim // 4, 1)
        )
        
        self.current_adas13 = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prev_dim // 4, 1)
        )
        
        self.current_mmse = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prev_dim // 4, 1)
        )
        
        # Future scores prediction (with time)
        future_input_dim = prev_dim + time_dim
        
        # Shared layers for future scores
        future_layers = []
        prev_dim = future_input_dim
        for hidden_dim in hidden_dims:
            future_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.future_shared = nn.Sequential(*future_layers)
        
        # Separate branches for future scores
        self.future_adas11 = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prev_dim // 4, 1)
        )
        
        self.future_adas13 = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prev_dim // 4, 1)
        )
        
        self.future_mmse = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prev_dim // 4, 1)
        )
        
        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()  # For metrics only
        
        # Metrics
        self.train_metrics = {
            'current_adas11_mse': [], 'current_adas13_mse': [], 'current_mmse_mse': [],
            'future_adas11_mse': [], 'future_adas13_mse': [], 'future_mmse_mse': [],
            'current_adas11_mae': [], 'current_adas13_mae': [], 'current_mmse_mae': [],
            'future_adas11_mae': [], 'future_adas13_mae': [], 'future_mmse_mae': [],
            'current_adas11_r2': [], 'current_adas13_r2': [], 'current_mmse_r2': [],
            'future_adas11_r2': [], 'future_adas13_r2': [], 'future_mmse_r2': []
        }
        self.val_metrics = {
            'current_adas11_mse': [], 'current_adas13_mse': [], 'current_mmse_mse': [],
            'future_adas11_mse': [], 'future_adas13_mse': [], 'future_mmse_mse': [],
            'current_adas11_mae': [], 'current_adas13_mae': [], 'current_mmse_mae': [],
            'future_adas11_mae': [], 'future_adas13_mae': [], 'future_mmse_mae': [],
            'current_adas11_r2': [], 'current_adas13_r2': [], 'current_mmse_r2': [],
            'future_adas11_r2': [], 'future_adas13_r2': [], 'future_mmse_r2': []
        }
        self.test_metrics = {
            'current_adas11_mse': [], 'current_adas13_mse': [], 'current_mmse_mse': [],
            'future_adas11_mse': [], 'future_adas13_mse': [], 'future_mmse_mse': [],
            'current_adas11_mae': [], 'current_adas13_mae': [], 'current_mmse_mae': [],
            'future_adas11_mae': [], 'future_adas13_mae': [], 'future_mmse_mae': [],
            'current_adas11_r2': [], 'current_adas13_r2': [], 'current_mmse_r2': [],
            'future_adas11_r2': [], 'future_adas13_r2': [], 'future_mmse_r2': []
        }
        
    def forward(self, mri, clinical, time_lapsed):
        """
        Forward pass
        
        Args:
            mri (torch.Tensor): 3D MRI image, shape (batch_size, 1, D, H, W)
            clinical (torch.Tensor): Clinical data, shape (batch_size, clinical_dim)
            time_lapsed (torch.Tensor): Number of days elapsed, shape (batch_size, 1)
            
        Returns:
            tuple: (current_scores, future_scores)
                current_scores: (adas11_now, adas13_now, mmse_now)
                future_scores: (adas11_future, adas13_future, mmse_future)
        """
        # Encode inputs
        mri_features = self.mri_encoder(mri)
        clinical_features = self.clinical_encoder(clinical)
        time_features = self.time_encoder(time_lapsed)
        
        # Current scores prediction
        current_features = torch.cat([mri_features, clinical_features], dim=1)
        current_shared = self.current_shared(current_features)
        
        current_adas11 = self.current_adas11(current_shared).squeeze(-1)
        current_adas13 = self.current_adas13(current_shared).squeeze(-1)
        current_mmse = self.current_mmse(current_shared).squeeze(-1)
        
        # Future scores prediction
        future_features = torch.cat([current_shared, time_features], dim=1)
        future_shared = self.future_shared(future_features)
        
        future_adas11 = self.future_adas11(future_shared).squeeze(-1)
        future_adas13 = self.future_adas13(future_shared).squeeze(-1)
        future_mmse = self.future_mmse(future_shared).squeeze(-1)
        
        current_scores = (current_adas11, current_adas13, current_mmse)
        future_scores = (future_adas11, future_adas13, future_mmse)
        
        return current_scores, future_scores
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate metrics for current and future predictions
        
        Args:
            predictions (tuple): (current_scores, future_scores)
            targets (tuple): (current_targets, future_targets)
            
        Returns:
            dict: Dictionary containing metrics
        """
        current_scores, future_scores = predictions
        current_targets, future_targets = targets
        
        metrics = {}
        
        # Current scores metrics
        for i, score in enumerate(['adas11', 'adas13', 'mmse']):
            pred = current_scores[i].detach().cpu().numpy()
            targ = current_targets[:, i].detach().cpu().numpy()
            
            metrics[f'current_{score}_mse'] = self.mse_criterion(current_scores[i], current_targets[:, i]).item()
            metrics[f'current_{score}_mae'] = self.mae_criterion(current_scores[i], current_targets[:, i]).item()
            metrics[f'current_{score}_r2'] = r2_score(targ, pred)
        
        # Future scores metrics
        for i, score in enumerate(['adas11', 'adas13', 'mmse']):
            pred = future_scores[i].detach().cpu().numpy()
            targ = future_targets[:, i].detach().cpu().numpy()
            
            metrics[f'future_{score}_mse'] = self.mse_criterion(future_scores[i], future_targets[:, i]).item()
            metrics[f'future_{score}_mae'] = self.mae_criterion(future_scores[i], future_targets[:, i]).item()
            metrics[f'future_{score}_r2'] = r2_score(targ, pred)
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Batch of data containing (mri, clinical, time_lapsed, targets)
            batch_idx: Batch index
            
        Returns:
            dict: Dictionary containing loss and metrics to log
        """
        mri, clinical, time_lapsed, targets = batch
        
        # Split targets into current and future
        current_targets = targets[:, :3]  # ADAS11_now, ADAS13_now, MMSCORE_now
        future_targets = targets[:, 3:]   # ADAS11_future, ADAS13_future, MMSCORE_future
        
        # Get predictions
        current_scores, future_scores = self(mri, clinical, time_lapsed)
        
        # Calculate losses
        current_losses = [
            self.mse_criterion(current_scores[0], current_targets[:, 0]),  # ADAS11
            self.mse_criterion(current_scores[1], current_targets[:, 1]),  # ADAS13
            self.mse_criterion(current_scores[2], current_targets[:, 2])   # MMSE
        ]
        
        future_losses = [
            self.mse_criterion(future_scores[0], future_targets[:, 0]),  # ADAS11
            self.mse_criterion(future_scores[1], future_targets[:, 1]),  # ADAS13
            self.mse_criterion(future_scores[2], future_targets[:, 2])   # MMSE
        ]
        
        # Total loss
        total_loss = sum(current_losses) + sum(future_losses)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            (current_scores, future_scores),
            (current_targets, future_targets)
        )
        
        # Log losses and metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, value in metrics.items():
            self.log(f'train_{metric_name}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.train_metrics[metric_name].append(value)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Batch of data containing (mri, clinical, time_lapsed, targets)
            batch_idx: Batch index
            
        Returns:
            dict: Dictionary containing loss and metrics to log
        """
        mri, clinical, time_lapsed, targets = batch
        
        # Split targets into current and future
        current_targets = targets[:, :3]  # ADAS11_now, ADAS13_now, MMSCORE_now
        future_targets = targets[:, 3:]   # ADAS11_future, ADAS13_future, MMSCORE_future
        
        # Get predictions
        current_scores, future_scores = self(mri, clinical, time_lapsed)
        
        # Calculate losses
        current_losses = [
            self.mse_criterion(current_scores[0], current_targets[:, 0]),  # ADAS11
            self.mse_criterion(current_scores[1], current_targets[:, 1]),  # ADAS13
            self.mse_criterion(current_scores[2], current_targets[:, 2])   # MMSE
        ]
        
        future_losses = [
            self.mse_criterion(future_scores[0], future_targets[:, 0]),  # ADAS11
            self.mse_criterion(future_scores[1], future_targets[:, 1]),  # ADAS13
            self.mse_criterion(future_scores[2], future_targets[:, 2])   # MMSE
        ]
        
        # Total loss
        total_loss = sum(current_losses) + sum(future_losses)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            (current_scores, future_scores),
            (current_targets, future_targets)
        )
        
        # Log losses and metrics
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.val_metrics[metric_name].append(value)
        
        return total_loss
    
    def on_train_epoch_end(self):
        """Callback called at the end of each training epoch"""
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in self.train_metrics.items()
        }
        
        # Print results
        print("\n" + "="*50)
        print(f"Epoch {self.current_epoch} - Training Metrics:")
        print("Current Scores:")
        print(f"ADAS11 - MSE: {avg_metrics['current_adas11_mse']:.4f}, MAE: {avg_metrics['current_adas11_mae']:.4f}, R2: {avg_metrics['current_adas11_r2']:.4f}")
        print(f"ADAS13 - MSE: {avg_metrics['current_adas13_mse']:.4f}, MAE: {avg_metrics['current_adas13_mae']:.4f}, R2: {avg_metrics['current_adas13_r2']:.4f}")
        print(f"MMSE - MSE: {avg_metrics['current_mmse_mse']:.4f}, MAE: {avg_metrics['current_mmse_mae']:.4f}, R2: {avg_metrics['current_mmse_r2']:.4f}")
        print("\nFuture Scores:")
        print(f"ADAS11 - MSE: {avg_metrics['future_adas11_mse']:.4f}, MAE: {avg_metrics['future_adas11_mae']:.4f}, R2: {avg_metrics['future_adas11_r2']:.4f}")
        print(f"ADAS13 - MSE: {avg_metrics['future_adas13_mse']:.4f}, MAE: {avg_metrics['future_adas13_mae']:.4f}, R2: {avg_metrics['future_adas13_r2']:.4f}")
        print(f"MMSE - MSE: {avg_metrics['future_mmse_mse']:.4f}, MAE: {avg_metrics['future_mmse_mae']:.4f}, R2: {avg_metrics['future_mmse_r2']:.4f}")
        print("="*50)
        
        # Reset metrics
        for metric in self.train_metrics:
            self.train_metrics[metric] = []
    
    def on_validation_epoch_end(self):
        """Callback called at the end of each validation epoch"""
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in self.val_metrics.items()
        }
        
        # Print results
        print(f"\nEpoch {self.current_epoch} - Validation Metrics:")
        print("Current Scores:")
        print(f"ADAS11 - MSE: {avg_metrics['current_adas11_mse']:.4f}, MAE: {avg_metrics['current_adas11_mae']:.4f}, R2: {avg_metrics['current_adas11_r2']:.4f}")
        print(f"ADAS13 - MSE: {avg_metrics['current_adas13_mse']:.4f}, MAE: {avg_metrics['current_adas13_mae']:.4f}, R2: {avg_metrics['current_adas13_r2']:.4f}")
        print(f"MMSE - MSE: {avg_metrics['current_mmse_mse']:.4f}, MAE: {avg_metrics['current_mmse_mae']:.4f}, R2: {avg_metrics['current_mmse_r2']:.4f}")
        print("\nFuture Scores:")
        print(f"ADAS11 - MSE: {avg_metrics['future_adas11_mse']:.4f}, MAE: {avg_metrics['future_adas11_mae']:.4f}, R2: {avg_metrics['future_adas11_r2']:.4f}")
        print(f"ADAS13 - MSE: {avg_metrics['future_adas13_mse']:.4f}, MAE: {avg_metrics['future_adas13_mae']:.4f}, R2: {avg_metrics['future_adas13_r2']:.4f}")
        print(f"MMSE - MSE: {avg_metrics['future_mmse_mse']:.4f}, MAE: {avg_metrics['future_mmse_mae']:.4f}, R2: {avg_metrics['future_mmse_r2']:.4f}")
        print("="*50)
        
        # Reset metrics
        for metric in self.val_metrics:
            self.val_metrics[metric] = []
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # AdamW optimizer with learning rate 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        
        # ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def test_step(self, batch, batch_idx):
        """
        Test step
        
        Args:
            batch: Batch of data containing (mri, clinical, time_lapsed, targets)
            batch_idx: Batch index
            
        Returns:
            dict: Dictionary containing loss and metrics to log
        """
        mri, clinical, time_lapsed, targets = batch
        
        # Split targets into current and future
        current_targets = targets[:, :3]  # ADAS11_now, ADAS13_now, MMSCORE_now
        future_targets = targets[:, 3:]   # ADAS11_future, ADAS13_future, MMSCORE_future
        
        # Get predictions
        current_scores, future_scores = self(mri, clinical, time_lapsed)
        
        # Calculate losses
        current_losses = [
            self.mse_criterion(current_scores[0], current_targets[:, 0]),  # ADAS11
            self.mse_criterion(current_scores[1], current_targets[:, 1]),  # ADAS13
            self.mse_criterion(current_scores[2], current_targets[:, 2])   # MMSE
        ]
        
        future_losses = [
            self.mse_criterion(future_scores[0], future_targets[:, 0]),  # ADAS11
            self.mse_criterion(future_scores[1], future_targets[:, 1]),  # ADAS13
            self.mse_criterion(future_scores[2], future_targets[:, 2])   # MMSE
        ]
        
        # Total loss
        total_loss = sum(current_losses) + sum(future_losses)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            (current_scores, future_scores),
            (current_targets, future_targets)
        )
        
        # Log losses and metrics
        self.log('test_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, value in metrics.items():
            self.log(f'test_{metric_name}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.test_metrics[metric_name].append(value)
        
        return total_loss
    
    def on_test_epoch_end(self):
        """Callback called at the end of each test epoch"""
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in self.test_metrics.items()
        }
        
        # Print results
        print(f"\nTest Results:")
        print("Current Scores:")
        print(f"ADAS11 - MSE: {avg_metrics['current_adas11_mse']:.4f}, MAE: {avg_metrics['current_adas11_mae']:.4f}, R2: {avg_metrics['current_adas11_r2']:.4f}")
        print(f"ADAS13 - MSE: {avg_metrics['current_adas13_mse']:.4f}, MAE: {avg_metrics['current_adas13_mae']:.4f}, R2: {avg_metrics['current_adas13_r2']:.4f}")
        print(f"MMSE - MSE: {avg_metrics['current_mmse_mse']:.4f}, MAE: {avg_metrics['current_mmse_mae']:.4f}, R2: {avg_metrics['current_mmse_r2']:.4f}")
        print("\nFuture Scores:")
        print(f"ADAS11 - MSE: {avg_metrics['future_adas11_mse']:.4f}, MAE: {avg_metrics['future_adas11_mae']:.4f}, R2: {avg_metrics['future_adas11_r2']:.4f}")
        print(f"ADAS13 - MSE: {avg_metrics['future_adas13_mse']:.4f}, MAE: {avg_metrics['future_adas13_mae']:.4f}, R2: {avg_metrics['future_adas13_r2']:.4f}")
        print(f"MMSE - MSE: {avg_metrics['future_mmse_mse']:.4f}, MAE: {avg_metrics['future_mmse_mae']:.4f}, R2: {avg_metrics['future_mmse_r2']:.4f}")
        
        # Reset metrics
        for metric in self.test_metrics:
            self.test_metrics[metric] = [] 