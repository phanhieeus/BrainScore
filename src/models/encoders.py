import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from torchinfo import summary


class MRIEncoder(nn.Module):
    def __init__(self, model_name="swinunetr", pretrained=True, freeze=True, feature_dim=512):
        """
        Initialize MRIEncoder to extract features from 3D MRI images using SwinUNETR
        
        Args:
            model_name (str): Name of model to use (currently only supports 'swinunetr')
            pretrained (bool): Whether to use pretrained weights
            freeze (bool): Whether to freeze the encoder backbone
            feature_dim (int): Dimension of output feature vector (default: 512)
        """
        super().__init__()
        self.encoder = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=1,
            feature_size=12,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True
        )
        
        # Remove the decoder part since we only need encoder features
        self.encoder.decoder = nn.Identity()
        self.encoder.out = nn.Identity()
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        
        # Initialize projection layer with None
        self.projection = None
            
        if freeze:
            print("Freezing encoder backbone")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing encoder backbone")
            for param in self.encoder.parameters():
                param.requires_grad = True

    def _initialize_projection(self, x):
        """Initialize projection layer based on encoder output size"""
        if self.projection is None:
            # Get feature size from encoder output
            features = self.encoder(x)
            features = self.pool(features)
            features = self.flatten(features)
            swin_feature_dim = features.shape[1]
            
            # Create projection layer on the same device as input
            self.projection = nn.Linear(swin_feature_dim, 512).to(x.device)
            print(f"Initialized projection layer: {swin_feature_dim} -> 512")

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, 1, D, H, W)
            
        Returns:
            torch.Tensor: Feature vector shape (batch_size, feature_dim)
        """
        # Initialize projection layer if not done yet
        self._initialize_projection(x)
        
        # Extract features from encoder
        features = self.encoder(x)
        
        # Global average pooling
        features = self.pool(features)
        
        # Flatten
        features = self.flatten(features)
        
        # Project to 512 dimensions
        features = self.projection(features)
        
        return features
    
    def get_feature_dim(self):
        """Return dimension of output feature vector"""
        return 512  # Always 512 as specified in __init__


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[256, 128, 64, 128, 256], output_dim=256, dropout_rate=0.1):
        """
        Initialize ClinicalEncoder to extract features from clinical data
        
        Args:
            input_dim (int): Dimension of input vector (3 demographic + 3 current scores + 1 time_lapsed + 1 diagnosis = 8)
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output feature vector
            dropout_rate (float): Dropout rate (default: 0.1)
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
                input_dim = 8 (PTGENDER, age, PTEDUCAT, ADAS11_now, ADAS13_now, MMSCORE_now, time_lapsed, DIAGNOSIS_now)
            
        Returns:
            torch.Tensor: Feature vector shape (batch_size, output_dim)
        """
        return self.encoder(x)
    
    def get_feature_dim(self):
        """Return dimension of output feature vector"""
        return self.encoder[-1].out_features 


if __name__ == "__main__":
    # Create sample inputs
    batch_size = 2
    mri_input = torch.randn(batch_size, 1, 96, 96, 96)  # MRI input
    clinical_input = torch.randn(batch_size, 8)  # Clinical input (8 features)
    
    # Initialize encoders
    mri_encoder = MRIEncoder(freeze=True)
    clinical_encoder = ClinicalEncoder()
    
    # Print summary for MRIEncoder
    print("\nMRIEncoder Summary:")
    print("=" * 50)
    summary(
        mri_encoder,
        input_size=(batch_size, 1, 96, 96, 96),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=4,
        device="cpu"
    )
    
    # Print summary for ClinicalEncoder
    print("\nClinicalEncoder Summary:")
    print("=" * 50)
    summary(
        clinical_encoder,
        input_size=(batch_size, 8),
        col_names=["input_size", "output_size", "num_params"],
        depth=4,
        device="cpu"
    )
    
    # Test forward pass
    print("\nTesting forward pass:")
    print("=" * 50)
    mri_features = mri_encoder(mri_input)
    clinical_features = clinical_encoder(clinical_input)
    
    print(f"MRI features shape: {mri_features.shape}")
    print(f"Clinical features shape: {clinical_features.shape}") 