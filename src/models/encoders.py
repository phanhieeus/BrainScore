import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from torchinfo import summary

# Constants for model configuration
DEFAULT_IMG_SIZE = (96, 96, 96)
DEFAULT_IN_CHANNELS = 1
DEFAULT_OUT_CHANNELS = 1
DEFAULT_FEATURE_SIZE = 12
DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_ATTN_DROPOUT_RATE = 0.0
DEFAULT_DROPOUT_PATH_RATE = 0.0
DEFAULT_FEATURE_DIM = 512

# Constants for clinical data
CLINICAL_INPUT_DIM = 8  # 3 demographic + 3 current scores + 1 time_lapsed + 1 diagnosis
DEFAULT_CLINICAL_HIDDEN_DIMS = [256, 128, 64, 128, 256]
DEFAULT_CLINICAL_OUTPUT_DIM = 256
DEFAULT_CLINICAL_DROPOUT_RATE = 0.1


class MRIEncoder(nn.Module):
    def __init__(self, hidden_dims=[128], output_dim=DEFAULT_FEATURE_DIM):
        """
        Initialize MRIEncoder to extract features from 3D MRI images using SwinUNETR
        
        Args:
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output feature vector (default: 512)
        """
        super().__init__()
        mri_encoder = SwinUNETR(img_size=DEFAULT_IMG_SIZE, in_channels=DEFAULT_IN_CHANNELS, out_channels=DEFAULT_OUT_CHANNELS)
        self.swin_backbone = mri_encoder.swinViT
        self.normalize = mri_encoder.normalize
        self.flatten = nn.Flatten()
        
        # Initialize reduce_conv layer based on dummy input
        dummy_tensor = torch.rand(1, DEFAULT_IN_CHANNELS, *DEFAULT_IMG_SIZE)
        hidden_state = self.swin_backbone(dummy_tensor, self.normalize)[-1]
        self.reduce_conv = nn.Conv3d(hidden_state.shape[1], 128, kernel_size=1, stride=1)
        
        # Initialize projection layers
        dummy_tensor = self.reduce_conv(hidden_state)
        layers = []
        prev_dim = self.flatten(dummy_tensor).shape[-1]
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
        self.output_dim = output_dim
        
        # Initialize weights
        self._init_weights()
        self._init_weights_swin()

    def _get_output_dim(self):
        return self.output_dim

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights_swin(self):
        for m in self.swin_backbone.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, 1, D, H, W)
            
        Returns:
            torch.Tensor: Feature vector shape (batch_size, output_dim)
        """
        x = self.swin_backbone(x, self.normalize)[-1]
        x = self.reduce_conv(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=CLINICAL_INPUT_DIM, hidden_dims=DEFAULT_CLINICAL_HIDDEN_DIMS, 
                 output_dim=DEFAULT_CLINICAL_OUTPUT_DIM):
        """
        Initialize ClinicalEncoder to extract features from clinical data
        
        Args:
            input_dim (int): Dimension of input vector (3 demographic + 3 current scores + 1 time_lapsed + 1 diagnosis = 8)
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output feature vector
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
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
        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_output_dim(self):
        return self.output_dim

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, input_dim)
                input_dim = 8 (PTGENDER, age, PTEDUCAT, ADAS11_now, ADAS13_now, MMSCORE_now, time_lapsed, DIAGNOSIS_now)
            
        Returns:
            torch.Tensor: Feature vector shape (batch_size, output_dim)
        """
        return self.projection(x)


if __name__ == "__main__":
    # Create sample inputs
    batch_size = 2
    mri_input = torch.randn(batch_size, DEFAULT_IN_CHANNELS, *DEFAULT_IMG_SIZE)  # MRI input
    clinical_input = torch.randn(batch_size, CLINICAL_INPUT_DIM)  # Clinical input
    
    # Initialize encoders
    mri_encoder = MRIEncoder(freeze=True)
    clinical_encoder = ClinicalEncoder()
    
    # Print summary for MRIEncoder
    print("\nMRIEncoder Summary:")
    print("=" * 50)
    summary(
        mri_encoder,
        input_size=(batch_size, DEFAULT_IN_CHANNELS, *DEFAULT_IMG_SIZE),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=4,
        device="cpu"
    )
    
    # Print summary for ClinicalEncoder
    print("\nClinicalEncoder Summary:")
    print("=" * 50)
    summary(
        clinical_encoder,
        input_size=(batch_size, CLINICAL_INPUT_DIM),
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