import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
import numpy as np

from .encoders import MRIEncoder, ClinicalEncoder, DEFAULT_FEATURE_DIM, DEFAULT_CLINICAL_OUTPUT_DIM

# Constants for model configuration
DEFAULT_HIDDEN_DIMS = [256, 512, 256, 128]
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_LEAKY_RELU_SLOPE = 0.2
DEFAULT_HEAD_HIDDEN_DIM = 64
NUM_SCORES = 3  # ADAS11, ADAS13, MMSE

# Constants for optimizer
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MIN_LR = 1e-6
DEFAULT_LR_PATIENCE = 5
DEFAULT_LR_FACTOR = 0.1


class FusionRegressor(pl.LightningModule):
    def __init__(self, mri_dim=DEFAULT_FEATURE_DIM, clinical_dim=DEFAULT_CLINICAL_OUTPUT_DIM, 
                 hidden_dims=DEFAULT_HIDDEN_DIMS, dropout_rate=DEFAULT_DROPOUT_RATE):
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
                nn.LeakyReLU(DEFAULT_LEAKY_RELU_SLOPE),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.fusion_network = nn.Sequential(*layers)
        
        # Future score prediction heads
        self.future_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], DEFAULT_HEAD_HIDDEN_DIM),
                nn.LeakyReLU(DEFAULT_LEAKY_RELU_SLOPE),
                nn.Dropout(dropout_rate),
                nn.Linear(DEFAULT_HEAD_HIDDEN_DIM, 1)
            ) for _ in range(NUM_SCORES)  # ADAS11, ADAS13, MMSE
        ])
        
        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        
        # Initialize metrics dictionaries with numpy arrays instead of lists
        metric_keys = []
        for score in ['adas11', 'adas13', 'mmse']:
            for metric in ['mse', 'mae', 'r2']:
                metric_keys.append(f'future_{score}_{metric}')
        
        # Use numpy arrays with fixed size to store metrics
        self.train_metrics = {key: np.zeros(100) for key in metric_keys}  # Store last 100 epochs
        self.val_metrics = {key: np.zeros(100) for key in metric_keys}
        self.test_metrics = {key: np.zeros(100) for key in metric_keys}
        self.metric_indices = {key: 0 for key in metric_keys}  # Track current position in arrays
        
        # Initialize running averages for metrics
        self.running_metrics = {key: 0.0 for key in metric_keys}
        self.running_counts = {key: 0 for key in metric_keys}
        
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
            self.mse_criterion(future_scores[i], targets[:, i]) for i in range(NUM_SCORES)
        ]
        
        # Total loss
        total_loss = sum(future_losses)
        
        # Calculate metrics
        metrics = self.calculate_metrics(future_scores, targets)
        
        # Log losses and metrics
        self.log(f'{stage}_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Update running averages
        for metric_name, value in metrics.items():
            # Log to tensorboard
            self.log(f'{stage}_{metric_name}', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            # Update running average
            self.running_metrics[metric_name] = (
                self.running_metrics[metric_name] * self.running_counts[metric_name] + value
            ) / (self.running_counts[metric_name] + 1)
            self.running_counts[metric_name] += 1
            
            # Store in numpy array at epoch end
            if batch_idx == 0:  # Start of new epoch
                metrics_dict = getattr(self, f'{stage}_metrics')
                idx = self.metric_indices[metric_name]
                metrics_dict[metric_name][idx] = self.running_metrics[metric_name]
                self.metric_indices[metric_name] = (idx + 1) % 100
                
                # Reset running averages
                self.running_metrics[metric_name] = 0.0
                self.running_counts[metric_name] = 0
        
        return total_loss
    
    def shared_epoch_end(self, stage='train'):
        """
        Shared epoch end for training, validation and testing
        
        Args:
            stage: Stage name ('train', 'val', or 'test')
        """
        # Get metrics for the last epoch
        metrics = getattr(self, f'{stage}_metrics')
        idx = self.metric_indices[list(metrics.keys())[0]] - 1
        if idx < 0:
            idx = 99
        
        # Print results for the last epoch only
        print("\n" + "="*50)
        print(f"Epoch {self.current_epoch} - {stage.capitalize()} Metrics:")
        print("Future Scores:")
        for score in ['adas11', 'adas13', 'mmse']:
            print(f"{score.capitalize()} - MSE: {metrics[f'future_{score}_mse'][idx]:.4f}, "
                  f"MAE: {metrics[f'future_{score}_mae'][idx]:.4f}, "
                  f"R2: {metrics[f'future_{score}_r2'][idx]:.4f}")
        print("="*50)
    
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