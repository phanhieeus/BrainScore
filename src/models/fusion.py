import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import numpy as np

from .encoders import MRIEncoder, ClinicalEncoder, TimeLapsedEncoder
from .interactions import SelfInteraction, CrossInteraction


class FusionRegressor(pl.LightningModule):
    def __init__(self, mri_dim=2048, clinical_dim=3, time_dim=64, hidden_dims=[512, 256], output_dim=4):
        """
        Model that combines information from MRI, clinical data, and time to predict 4 values
        
        Args:
            mri_dim (int): Dimension of MRI feature vector
            clinical_dim (int): Dimension of clinical data vector
            time_dim (int): Dimension of time feature vector
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Number of values to predict (default is 4)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders
        self.mri_encoder = MRIEncoder(feature_dim=mri_dim)
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_dim)
        self.time_encoder = TimeLapsedEncoder(output_dim=time_dim)
        
        # Interaction layers
        self.self_interaction = SelfInteraction(input_dim=self.clinical_encoder.get_feature_dim())
        self.cross_interaction = CrossInteraction(dim1=mri_dim, dim2=self.self_interaction.get_output_dim())
        
        # Fusion layers
        fusion_input_dim = self.cross_interaction.get_output_dim() + time_dim
        
        layers = []
        prev_dim = fusion_input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.fusion_layers = nn.Sequential(*layers)
        
        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        
        # Metrics
        self.train_metrics = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        self.val_metrics = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
    def forward(self, mri, clinical, time_lapsed):
        """
        Forward pass
        
        Args:
            mri (torch.Tensor): 3D MRI image, shape (batch_size, 1, D, H, W)
            clinical (torch.Tensor): Clinical data, shape (batch_size, clinical_dim)
            time_lapsed (torch.Tensor): Number of days elapsed, shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Predictions of 4 values, shape (batch_size, 4)
        """
        # Encode MRI
        mri_features = self.mri_encoder(mri)  # (batch_size, mri_dim)
        
        # Encode clinical data
        clinical_features = self.clinical_encoder(clinical)  # (batch_size, clinical_dim)
        
        # Self-interaction for clinical features
        clinical_interacted = self.self_interaction(clinical_features)  # (batch_size, clinical_interacted_dim)
        
        # Cross-interaction between MRI and clinical features
        cross_features = self.cross_interaction(mri_features, clinical_interacted)  # (batch_size, cross_dim)
        
        # Encode time
        time_features = self.time_encoder(time_lapsed)  # (batch_size, time_dim)
        
        # Combine cross features and time features
        combined_features = torch.cat([cross_features, time_features], dim=1)
        
        # Predict
        predictions = self.fusion_layers(combined_features)
        
        return predictions
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate metrics
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            dict: Dictionary containing metrics
        """
        # Convert to numpy for R2 score calculation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Calculate R2 score for each output
        r2_scores = []
        for i in range(pred_np.shape[1]):
            r2 = r2_score(target_np[:, i], pred_np[:, i])
            r2_scores.append(r2)
        
        return {
            'mse': self.mse_criterion(predictions, targets).item(),
            'mae': self.mae_criterion(predictions, targets).item(),
            'r2': np.mean(r2_scores)  # Average R2 score across outputs
        }
    
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
        predictions = self(mri, clinical, time_lapsed)
        
        # Calculate loss
        mse_loss = self.mse_criterion(predictions, targets)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Log loss and metrics
        self.log('train_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse', metrics['mse'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', metrics['mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_r2', metrics['r2'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Save metrics for epoch end
        for metric_name, value in metrics.items():
            self.train_metrics[metric_name].append(value)
        
        return mse_loss
    
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
        predictions = self(mri, clinical, time_lapsed)
        
        # Calculate loss
        mse_loss = self.mse_criterion(predictions, targets)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Log loss and metrics
        self.log('val_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', metrics['mse'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', metrics['mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2', metrics['r2'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Save metrics for epoch end
        for metric_name, value in metrics.items():
            self.val_metrics[metric_name].append(value)
        
        return mse_loss
    
    def on_train_epoch_end(self):
        """
        Callback called at the end of each training epoch
        """
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in self.train_metrics.items()
        }
        
        # Print results
        print(f"\nEpoch {self.current_epoch} - Training Metrics:")
        print(f"Loss: {avg_metrics['mse']:.4f}")
        print(f"MSE: {avg_metrics['mse']:.4f}")
        print(f"MAE: {avg_metrics['mae']:.4f}")
        print(f"R2 Score: {avg_metrics['r2']:.4f}")
        
        # Reset metrics
        for metric in self.train_metrics:
            self.train_metrics[metric] = []
    
    def on_validation_epoch_end(self):
        """
        Callback called at the end of each validation epoch
        """
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in self.val_metrics.items()
        }
        
        # Print results
        print(f"\nEpoch {self.current_epoch} - Validation Metrics:")
        print(f"Loss: {avg_metrics['mse']:.4f}")
        print(f"MSE: {avg_metrics['mse']:.4f}")
        print(f"MAE: {avg_metrics['mae']:.4f}")
        print(f"R2 Score: {avg_metrics['r2']:.4f}")
        
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