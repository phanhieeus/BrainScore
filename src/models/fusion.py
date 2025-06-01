import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import numpy as np

from .encoders import MRIEncoder, ClinicalEncoder, TimeLapsedEncoder


class FusionRegressor(pl.LightningModule):
    def __init__(self, mri_dim=2048, clinical_dim=6, time_dim=256, hidden_dims=[256, 512, 1024, 512, 256], MRI_encoder_freeze=True):
        """
        Model that combines information from MRI, clinical data, and time to predict future scores
        
        Args:
            mri_dim (int): Dimension of MRI feature vector
            clinical_dim (int): Dimension of clinical data vector (3 demographic + 3 current scores)
            time_dim (int): Dimension of time feature vector
            hidden_dims (list): List of hidden layer dimensions
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders
        self.mri_encoder = MRIEncoder(feature_dim=mri_dim, freeze=MRI_encoder_freeze)
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_dim)
        self.time_encoder = TimeLapsedEncoder(output_dim=time_dim)
        
        # Future scores prediction
        future_input_dim = mri_dim + self.clinical_encoder.get_feature_dim() + time_dim
        
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
            'future_adas11_mse': [], 'future_adas13_mse': [], 'future_mmse_mse': [],
            'future_adas11_mae': [], 'future_adas13_mae': [], 'future_mmse_mae': [],
            'future_adas11_r2': [], 'future_adas13_r2': [], 'future_mmse_r2': []
        }
        self.val_metrics = {
            'future_adas11_mse': [], 'future_adas13_mse': [], 'future_mmse_mse': [],
            'future_adas11_mae': [], 'future_adas13_mae': [], 'future_mmse_mae': [],
            'future_adas11_r2': [], 'future_adas13_r2': [], 'future_mmse_r2': []
        }
        self.test_metrics = {
            'future_adas11_mse': [], 'future_adas13_mse': [], 'future_mmse_mse': [],
            'future_adas11_mae': [], 'future_adas13_mae': [], 'future_mmse_mae': [],
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
            tuple: (adas11_future, adas13_future, mmse_future)
        """
        # Encode inputs
        mri_features = self.mri_encoder(mri)
        clinical_features = self.clinical_encoder(clinical)
        time_features = self.time_encoder(time_lapsed)
        
        # Combine all features
        combined_features = torch.cat([mri_features, clinical_features, time_features], dim=1)
        
        # Future scores prediction
        future_shared = self.future_shared(combined_features)
        
        future_adas11 = self.future_adas11(future_shared).squeeze(-1)
        future_adas13 = self.future_adas13(future_shared).squeeze(-1)
        future_mmse = self.future_mmse(future_shared).squeeze(-1)
        
        return future_adas11, future_adas13, future_mmse
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate metrics for future predictions
        
        Args:
            predictions (tuple): (adas11_future, adas13_future, mmse_future)
            targets (torch.Tensor): Future targets shape (batch_size, 3)
            
        Returns:
            dict: Dictionary containing metrics
        """
        metrics = {}
        
        # Future scores metrics
        for i, score in enumerate(['adas11', 'adas13', 'mmse']):
            pred = predictions[i].detach().cpu().numpy()
            targ = targets[:, i].detach().cpu().numpy()
            
            metrics[f'future_{score}_mse'] = self.mse_criterion(predictions[i], targets[:, i]).item()
            metrics[f'future_{score}_mae'] = self.mae_criterion(predictions[i], targets[:, i]).item()
            metrics[f'future_{score}_r2'] = r2_score(targ, pred)
        
        return metrics
    
    def shared_step(self, batch, batch_idx, stage='train'):
        """
        Shared step for training, validation and testing
        
        Args:
            batch: Batch of data containing (mri, clinical, time_lapsed, targets)
            batch_idx: Batch index
            stage: Stage name ('train', 'val', or 'test')
            
        Returns:
            dict: Dictionary containing loss and metrics
        """
        mri, clinical, time_lapsed, targets = batch
        
        # Get future targets
        future_targets = targets[:, 3:]   # ADAS11_future, ADAS13_future, MMSCORE_future
        
        # Get predictions
        future_scores = self(mri, clinical, time_lapsed)
        
        # Calculate losses
        future_losses = [
            self.mse_criterion(future_scores[0], future_targets[:, 0]),  # ADAS11
            self.mse_criterion(future_scores[1], future_targets[:, 1]),  # ADAS13
            self.mse_criterion(future_scores[2], future_targets[:, 2])   # MMSE
        ]
        
        # Total loss
        total_loss = sum(future_losses)
        
        # Calculate metrics
        metrics = self.calculate_metrics(future_scores, future_targets)
        
        # Log losses and metrics
        self.log(f'{stage}_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, value in metrics.items():
            self.log(f'{stage}_{metric_name}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            getattr(self, f'{stage}_metrics')[metric_name].append(value)
        
        return total_loss
    
    def shared_epoch_end(self, stage='train'):
        """
        Shared epoch end for training, validation and testing
        
        Args:
            stage: Stage name ('train', 'val', or 'test')
        """
        # Calculate average metrics
        metrics = getattr(self, f'{stage}_metrics')
        avg_metrics = {
            metric: np.mean(values) for metric, values in metrics.items()
        }
        
        # Print results
        print("\n" + "="*50)
        print(f"Epoch {self.current_epoch} - {stage.capitalize()} Metrics:")
        print("Future Scores:")
        print(f"ADAS11 - MSE: {avg_metrics['future_adas11_mse']:.4f}, MAE: {avg_metrics['future_adas11_mae']:.4f}, R2: {avg_metrics['future_adas11_r2']:.4f}")
        print(f"ADAS13 - MSE: {avg_metrics['future_adas13_mse']:.4f}, MAE: {avg_metrics['future_adas13_mae']:.4f}, R2: {avg_metrics['future_adas13_r2']:.4f}")
        print(f"MMSE - MSE: {avg_metrics['future_mmse_mse']:.4f}, MAE: {avg_metrics['future_mmse_mae']:.4f}, R2: {avg_metrics['future_mmse_r2']:.4f}")
        print("="*50)
        
        # Reset metrics
        for metric in metrics:
            metrics[metric] = []
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        return self.shared_step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        return self.shared_step(batch, batch_idx, stage='val')
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        return self.shared_step(batch, batch_idx, stage='test')
    
    def on_train_epoch_end(self):
        """Callback called at the end of each training epoch"""
        self.shared_epoch_end(stage='train')
    
    def on_validation_epoch_end(self):
        """Callback called at the end of each validation epoch"""
        self.shared_epoch_end(stage='val')
    
    def on_test_epoch_end(self):
        """Callback called at the end of each test epoch"""
        self.shared_epoch_end(stage='test')
    
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