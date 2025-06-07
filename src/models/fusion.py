import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import numpy as np

from .encoders import MRIEncoder

# Constants for model configuration
DEFAULT_CLINIC_INPUT_DIM = 8  # 3 demographic + 3 current scores + 1 time_lapsed + 1 diagnosis
DEFAULT_CLINIC_HIDDEN_DIM = 32
DEFAULT_CLINIC_OUTPUT_DIM = 64
DEFAULT_FUSION_HIDDEN_DIM = 128

# Constants for optimizer
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MIN_LR = 1e-6
DEFAULT_LR_PATIENCE = 5
DEFAULT_LR_FACTOR = 0.1


class FusionRegressor(pl.LightningModule):
    def __init__(self, 
                 clinic_input_dim=DEFAULT_CLINIC_INPUT_DIM,
                 clinic_hidden_dim=DEFAULT_CLINIC_HIDDEN_DIM,
                 clinic_output_dim=DEFAULT_CLINIC_OUTPUT_DIM,
                 fusion_hidden_dim=DEFAULT_FUSION_HIDDEN_DIM,
                 pretrained=True,
                 freeze=True):
        """
        Model that combines information from MRI and clinical data to predict future scores
        
        Args:
            clinic_input_dim (int): Dimension of clinical input
            clinic_hidden_dim (int): Hidden dimension for clinical features
            clinic_output_dim (int): Output dimension for clinical features
            fusion_hidden_dim (int): Hidden dimension for fusion network
            pretrained (bool): Whether to use pretrained weights for MRI encoder
            freeze (bool): Whether to freeze MRI encoder weights
        """
        super().__init__()
        self.save_hyperparameters()
        
        # MRI Encoder and head
        self.mri_encoder = MRIEncoder(pretrained=pretrained, freeze=freeze)
        self.mri_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.LeakyReLU(0.01),
            nn.Linear(384, 192),
            nn.LeakyReLU(0.01)
        )
        
        # Clinical features processing
        self.future_demo = nn.Sequential(
            nn.Linear(clinic_input_dim, clinic_hidden_dim),
            nn.ReLU(),
            nn.Linear(clinic_hidden_dim, clinic_output_dim)
        )
        
        # Prediction heads
        self.adas11_future = nn.Sequential(
            nn.Linear(192 + clinic_output_dim, fusion_hidden_dim),
            nn.Softplus(),
            nn.Linear(fusion_hidden_dim, 1)
        )
        
        self.adas13_future = nn.Sequential(
            nn.Linear(192 + clinic_output_dim, fusion_hidden_dim),
            nn.Softplus(),
            nn.Linear(fusion_hidden_dim, 1)
        )
        
        self.mmscore_future = nn.Sequential(
            nn.Linear(192 + clinic_output_dim, fusion_hidden_dim),
            nn.Softplus(),
            nn.Linear(fusion_hidden_dim, 1)
        )
        
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

    def forward(self, mri, clinical, time_lapse, y_now):
        """
        Forward pass
        
        Args:
            mri (torch.Tensor): 3D MRI image, shape (batch_size, 1, D, H, W)
            clinical (torch.Tensor): Clinical data, shape (batch_size, 8)
            time_lapse (torch.Tensor): Time lapse, shape (batch_size, 1)
            y_now (torch.Tensor): Current scores, shape (batch_size, 3)
            
        Returns:
            tuple: (adas11_future, adas13_future, mmse_future)
        """
        # Process MRI features
        mri_features = self.mri_encoder(mri)
        mri_features = self.mri_head(mri_features)
        
        # Process clinical features
        future_features = torch.cat((clinical, time_lapse, y_now), dim=1)
        future_features = self.future_demo(future_features)
        
        # Fuse features and make predictions
        fused_features = torch.cat((mri_features, future_features), dim=1)
        adas11_future = self.adas11_future(fused_features)
        adas13_future = self.adas13_future(fused_features)
        mmscore_future = self.mmscore_future(fused_features)
        
        return adas11_future, adas13_future, mmscore_future

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
        future_scores = self(mri, clinical, torch.zeros(mri.shape[0], 1), torch.zeros(mri.shape[0], 3))
        
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