import torch
import torch.nn as nn
from monai.networks.nets import ResNetFeatures


class MRIEncoder(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True, freeze=True, feature_dim=2048):
        """
        Initialize MRIEncoder to extract features from 3D MRI images
        
        Args:
            model_name (str): Name of ResNet model to use ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained (bool): Whether to use pretrained weights
            freeze (bool): Whether to freeze the ResNet backbone
            feature_dim (int): Dimension of output feature vector
        """
        super().__init__()
        self.encoder = ResNetFeatures(
            model_name=model_name,
            pretrained=pretrained,
            spatial_dims=3,
            in_channels=1
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        
        # Get feature dimension from ResNet
        if model_name == 'resnet18' or model_name == 'resnet34':
            self.resnet_feature_dim = 512
        else:  # resnet50, resnet101, resnet152
            self.resnet_feature_dim = 2048
            
        # Projection layer to adjust feature dimension if needed
        if feature_dim != self.resnet_feature_dim:
            self.projection = nn.Linear(self.resnet_feature_dim, feature_dim)
        else:
            self.projection = nn.Identity()
            
        if freeze:
            print("Freezing encoder backbone")
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, 1, D, H, W)
            
        Returns:
            torch.Tensor: Feature vector shape (batch_size, feature_dim)
        """
        # Extract features from encoder
        features = self.encoder(x)
        
        # ResNetFeatures returns a list of features, take the last one
        if isinstance(features, list):
            features = features[-1]
        
        # Global average pooling
        features = self.pool(features)
        
        # Flatten
        features = self.flatten(features)
        
        # Projection if needed
        features = self.projection(features)
        
        return features
    
    def get_feature_dim(self):
        """Return dimension of output feature vector"""
        return self.resnet_feature_dim if isinstance(self.projection, nn.Identity) else self.projection.out_features


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[128], output_dim=64, dropout_rate=0.2):
        """
        Initialize ClinicalEncoder to extract features from clinical data
        
        Args:
            input_dim (int): Dimension of input vector
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output feature vector
            dropout_rate (float): Dropout rate
        """
        super().__init__()
        
        # Create list of layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Feature vector shape (batch_size, output_dim)
        """
        return self.encoder(x)
    
    def get_feature_dim(self):
        """Return dimension of output feature vector"""
        return self.encoder[-1].out_features


class TimeLapsedEncoder(nn.Module):
    def __init__(self, output_dim=64, time_scale=30.0):
        """
        Time elapsed encoder, combining both linear and non-linear representations
        
        Args:
            output_dim (int): Dimension of output feature vector
            time_scale (float): Scaling factor for time normalization (default is 365 days)
        """
        super().__init__()
        self.time_scale = time_scale
        
        # Linear transformation
        self.linear = nn.Sequential(
            nn.Linear(1, output_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Non-linear (sinusoidal) transformation
        self.nonlinear = nn.Sequential(
            nn.Linear(1, output_dim // 2),
            nn.Sigmoid(),  # Scale to [0,1]
            nn.Linear(output_dim // 2, output_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, time_lapsed):
        """
        Forward pass
        
        Args:
            time_lapsed (torch.Tensor): Number of days elapsed, shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Time feature vector, shape (batch_size, output_dim)
        """
        # Normalize time
        normalized_time = time_lapsed / self.time_scale
        
        # Linear transformation
        linear_features = self.linear(normalized_time)
        
        # Non-linear transformation with sine function
        sin_time = torch.sin(normalized_time * 2 * torch.pi)  # 1-month period
        nonlinear_features = self.nonlinear(sin_time)
        
        # Combine both types of features
        return torch.cat([linear_features, nonlinear_features], dim=1)
    
    def get_feature_dim(self):
        """Return dimension of output feature vector"""
        return self.linear[0].out_features + self.nonlinear[-2].out_features 