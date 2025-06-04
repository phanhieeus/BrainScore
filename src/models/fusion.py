import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import numpy as np

from .encoders import MRIEncoder, ClinicalEncoder, DEFAULT_FEATURE_DIM, DEFAULT_CLINICAL_OUTPUT_DIM

# Constants for model configuration
DEFAULT_HIDDEN_DIMS = [256]
DEFAULT_OUTPUT_DIM = 512
DEFAULT_ADAS11_HIDDEN_DIMS = [256, 32]
DEFAULT_ADAS13_HIDDEN_DIMS = [256, 32]
DEFAULT_MMSCORE_HIDDEN_DIMS = [256, 128, 64]

# Constants for optimizer
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MIN_LR = 1e-6
DEFAULT_LR_PATIENCE = 5
DEFAULT_LR_FACTOR = 0.1


class FusionRegressor(pl.LightningModule):
    def __init__(self, hidden_dims=DEFAULT_HIDDEN_DIMS, output_dim=DEFAULT_OUTPUT_DIM,
                 ADAS11_hidden_dims=DEFAULT_ADAS11_HIDDEN_DIMS,
                 ADAS13_hidden_dims=DEFAULT_ADAS13_HIDDEN_DIMS,
                 MMSCORE_hidden_dims=DEFAULT_MMSCORE_HIDDEN_DIMS):
        """
        Model that combines information from MRI and clinical data to predict future scores
        
        Args:
            hidden_dims (list): List of hidden dimensions for fusion network
            output_dim (int): Output dimension of fusion network
            ADAS11_hidden_dims (list): List of hidden dimensions for ADAS11 head
            ADAS13_hidden_dims (list): List of hidden dimensions for ADAS13 head
            MMSCORE_hidden_dims (list): List of hidden dimensions for MMSE head
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders
        self.mri_encoder = MRIEncoder(hidden_dims=[])
        self.clinical_encoder = ClinicalEncoder()
        
        # Fusion network
        prev_dim = self.mri_encoder._get_output_dim() + self.clinical_encoder._get_output_dim()
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.projection = nn.Sequential(*layers)
        
        # ADAS11 head
        prev_dim = output_dim
        layers = []
        for hidden_dim in ADAS11_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.ADAS11_head = nn.Sequential(*layers)
        
        # ADAS13 head
        prev_dim = output_dim
        layers = []
        for hidden_dim in ADAS13_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.ADAS13_head = nn.Sequential(*layers)
        
        # MMSE head
        prev_dim = output_dim
        layers = []
        for hidden_dim in MMSCORE_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.MMSCORE_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        
        # Initialize metrics for epoch averages
        self.train_metrics = {key: 0.0 for key in ['adas11_mse', 'adas11_mae', 'adas11_r2',
                                                  'adas13_mse', 'adas13_mae', 'adas13_r2',
                                                  'mmse_mse', 'mmse_mae', 'mmse_r2']}
        self.val_metrics = {key: 0.0 for key in self.train_metrics.keys()}
        self.train_counts = {key: 0 for key in self.train_metrics.keys()}
        self.val_counts = {key: 0 for key in self.val_metrics.keys()}

    def _init_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_mri_encoder(self):
        return self.mri_encoder

    def _get_clinical_encoder(self):
        return self.clinical_encoder

    def forward(self, mri, clinical):
        """
        Forward pass
        
        Args:
            mri (torch.Tensor): 3D MRI image, shape (batch_size, 1, D, H, W)
            clinical (torch.Tensor): Clinical data, shape (batch_size, 8)
            
        Returns:
            tuple: (adas11_future, adas13_future, mmse_future)
        """
        mri_encoded = self.mri_encoder(mri)
        clinical_encoded = self.clinical_encoder(clinical)
        x = torch.cat([mri_encoded, clinical_encoded], dim=1)
        x = self.projection(x)
        ADAS11 = self.ADAS11_head(x)
        ADAS13 = self.ADAS13_head(x)
        MMSCORE = self.MMSCORE_head(x)
        return ADAS11, ADAS13, MMSCORE

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
            
            metrics[f'{score}_mse'] = self.mse_criterion(predictions[i], targets[:, i]).item()
            metrics[f'{score}_mae'] = self.mae_criterion(predictions[i], targets[:, i]).item()
            metrics[f'{score}_r2'] = r2_score(targ, pred)
        
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
        
        # Update running averages
        metrics_dict = getattr(self, f'{stage}_metrics')
        counts_dict = getattr(self, f'{stage}_counts')
        
        for metric_name, value in metrics.items():
            metrics_dict[metric_name] = (metrics_dict[metric_name] * counts_dict[metric_name] + value) / (counts_dict[metric_name] + 1)
            counts_dict[metric_name] += 1
        
        return total_loss

    def shared_epoch_end(self, stage='train'):
        """
        Shared epoch end for training, validation and testing
        
        Args:
            stage: Stage name ('train', 'val', or 'test')
        """
        metrics = getattr(self, f'{stage}_metrics')
        
        print(f"\nEpoch {self.current_epoch} - {stage.capitalize()} Metrics:")
        print("=" * 50)
        for score in ['adas11', 'adas13', 'mmse']:
            print(f"{score.upper()}:")
            print(f"  MSE: {metrics[f'{score}_mse']:.4f}")
            print(f"  MAE: {metrics[f'{score}_mae']:.4f}")
            print(f"  R2:  {metrics[f'{score}_r2']:.4f}")
        print("=" * 50)
        
        # Reset metrics for next epoch
        for key in metrics.keys():
            metrics[key] = 0.0
            getattr(self, f'{stage}_counts')[key] = 0

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=DEFAULT_LEARNING_RATE)
        
        # ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=DEFAULT_LR_FACTOR,
            patience=DEFAULT_LR_PATIENCE,
            min_lr=DEFAULT_MIN_LR
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        } 