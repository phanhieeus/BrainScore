import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from torchinfo import summary


class MRIEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        """
        Initialize MRIEncoder to extract features from 3D MRI images using SwinUNETR
        
        Args:
            pretrained (bool): Whether to use pretrained weights
            freeze (bool): Whether to freeze the backbone weights
        """
        super().__init__()
        input_size = (64, 64, 64)  # Input patch size
        in_channels = 1            # Number of input channels
        out_channels = 1           # Number of output channels (not important for encoder)
        spatial_dims = 3           # Number of spatial dimensions
        feature_size = 48          # Feature map size
        depths = [3, 3, 3, 3]      # Number of Swin Transformer blocks in each stage
        num_heads = [3, 6, 12, 24] # Number of attention heads

        self.encoder = SwinUNETR(
            img_size=input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            spatial_dims=spatial_dims
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.output_dim = 768  # Output dimension from SwinUNETR

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _get_output_dim(self):
        return self.output_dim

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, 1, D, H, W)
            
        Returns:
            torch.Tensor: Feature vector shape (batch_size, output_dim)
        """
        features = self.encoder.swinViT(x)[-1]
        pooled = self.pool(features).view(features.size(0), -1)
        return pooled


if __name__ == "__main__":
    # Create sample inputs
    batch_size = 2
    mri_input = torch.randn(batch_size, 1, 64, 64, 64)  # MRI input
    
    # Initialize encoder
    mri_encoder = MRIEncoder(freeze=True)
    
    # Print summary for MRIEncoder
    print("\nMRIEncoder Summary:")
    print("=" * 50)
    summary(
        mri_encoder,
        input_size=(batch_size, 1, 64, 64, 64),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=4,
        device="cpu"
    )
    
    # Test forward pass
    print("\nTesting forward pass:")
    print("=" * 50)
    mri_features = mri_encoder(mri_input)
    print(f"MRI features shape: {mri_features.shape}") 