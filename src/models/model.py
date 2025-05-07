import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from typing import Dict, List, Tuple

class BrainScoreModel(nn.Module):
    def __init__(
        self,
        num_demographic_features: int = 3,
        num_cognitive_scores: int = 4,
        dropout_rate: float = 0.3
    ):
        """
        Multi-input model for predicting cognitive scores from MRI and demographic data.
        
        Args:
            num_demographic_features: Number of demographic features
            num_cognitive_scores: Number of cognitive scores to predict
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # MRI feature extractor (using pre-trained DenseNet121)
        self.mri_encoder = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1024
        )
        
        # Demographic feature processor
        self.demographic_processor = nn.Sequential(
            nn.Linear(num_demographic_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined feature processor
        self.combined_processor = nn.Sequential(
            nn.Linear(1024 + 32, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_cognitive_scores)
        )
        
    def forward(self, image: torch.Tensor, demographic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            image: MRI image tensor of shape (batch_size, 1, height, width, depth)
            demographic: Demographic features tensor of shape (batch_size, num_demographic_features)
            
        Returns:
            Predicted cognitive scores tensor of shape (batch_size, num_cognitive_scores)
        """
        # Extract MRI features
        mri_features = self.mri_encoder(image)
        
        # Process demographic features
        demographic_features = self.demographic_processor(demographic)
        
        # Combine features
        combined_features = torch.cat([mri_features, demographic_features], dim=1)
        
        # Predict cognitive scores
        cognitive_scores = self.combined_processor(combined_features)
        
        return cognitive_scores
    
    def get_mri_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get MRI features for Grad-CAM visualization.
        
        Args:
            image: MRI image tensor
            
        Returns:
            MRI features tensor
        """
        return self.mri_encoder(image) 