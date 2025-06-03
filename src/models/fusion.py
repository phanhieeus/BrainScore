import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import numpy as np

from .encoders import MRIEncoder, ClinicalEncoder


class FusionRegressor(pl.LightningModule):
    def __init__(self, mri_dim=512, clinical_dim=256, hidden_dims=[256, 512, 256, 128], dropout_rate=0.1):
        """
        Model that combines information from MRI and clinical data to predict future scores
        
        Args:
            mri_dim (int): Output dimension of MRI encoder (default: 512)
            clinical_dim (int): Output dimension of clinical encoder (default: 256)
            hidden_dims (list): List of hidden dimensions for fusion network
            dropout_rate (float): Dropout rate for regularization (default: 0.1)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders
        self.mri_encoder = MRIEncoder(feature_dim=mri_dim, freeze=False)
        self.clinical_encoder = ClinicalEncoder(output_dim=clinical_dim, dropout_rate=dropout_rate)
        
        # Fusion network
        fusion_input_dim = mri_dim + clinical_dim
        layers = []
        prev_dim = fusion_input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.fusion_network = nn.Sequential(*layers)
        
        # Future score prediction heads
        self.future_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1)
            ) for _ in range(3)  # ADAS11, ADAS13, MMSE
        ])
        
        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        
        # Initialize metrics dictionaries
        metric_keys = []
        for score in ['adas11', 'adas13', 'mmse']:
            for metric in ['mse', 'mae', 'r2']:
                metric_keys.append(f'future_{score}_{metric}')
        
        self.train_metrics = {key: [] for key in metric_keys}
        self.val_metrics = {key: [] for key in metric_keys}
        self.test_metrics = {key: [] for key in metric_keys}
        
    def forward(self, mri, clinical):
        """
        Forward pass
        
        Args:
            mri (torch.Tensor): 3D MRI image, shape (batch_size, 1, D, H, W)
            clinical (torch.Tensor): Clinical data, shape (batch_size, 8)
            
        Returns:
            tuple: (adas11_future, adas13_future, mmse_future)
        """
        # Encode inputs
        mri_features = self.mri_encoder(mri)
        clinical_features = self.clinical_encoder(clinical)
        
        # Combine features
        combined_features = torch.cat([mri_features, clinical_features], dim=1)
        
        # Process through fusion network
        shared_features = self.fusion_network(combined_features)
        
        # Get predictions for each score
        future_scores = [head(shared_features).squeeze(-1) for head in self.future_heads]
        
        return future_scores
    
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
            batch: Batch of data containing (mri, clinical, targets)
            batch_idx: Batch index
            stage: Stage name ('train', 'val', or 'test')
            
        Returns:
            dict: Dictionary containing loss and metrics
        """
        mri, clinical, targets = batch
        
        # Get predictions
        future_scores = self(mri, clinical)
        
        # Calculate losses
        future_losses = [
            self.mse_criterion(future_scores[i], targets[:, i]) for i in range(3)
        ]
        
        # Total loss
        total_loss = sum(future_losses)
        
        # Calculate metrics
        metrics = self.calculate_metrics(future_scores, targets)
        
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
        for score in ['adas11', 'adas13', 'mmse']:
            print(f"{score.capitalize()} - MSE: {avg_metrics[f'future_{score}_mse']:.4f}, MAE: {avg_metrics[f'future_{score}_mae']:.4f}, R2: {avg_metrics[f'future_{score}_r2']:.4f}")
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