import torch
import torch.nn as nn
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.encoders import MRIEncoder
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

def print_encoder_architecture(encoder):
    """
    Print detailed architecture of MRIEncoder including layer names and parameters
    
    Args:
        encoder (MRIEncoder): MRIEncoder instance
    """
    print("\n=== MRIEncoder Architecture ===")
    print(f"Feature dimension: {encoder.get_feature_dim()}")
    print(f"Projection layer: {encoder.projection}")
    
    print("\n=== ResNet Backbone Layers ===")
    for name, module in encoder.encoder.named_modules():
        if isinstance(module, (nn.Conv3d, nn.BatchNorm3d, nn.Linear)):
            print(f"\nLayer: {name}")
            print(f"Type: {module.__class__.__name__}")
            if isinstance(module, nn.Conv3d):
                print(f"Parameters: {module.weight.shape}")
                print(f"Bias: {module.bias is not None}")
            elif isinstance(module, nn.BatchNorm3d):
                print(f"Parameters: {module.weight.shape}")
                print(f"Running stats: {module.running_mean.shape}")
            elif isinstance(module, nn.Linear):
                print(f"Parameters: {module.weight.shape}")
                print(f"Bias: {module.bias is not None}")

def print_encoder_summary(encoder, input_size=(1, 1, 64, 64, 64)):
    """
    Print detailed summary of encoder using torchinfo
    
    Args:
        encoder (MRIEncoder): MRIEncoder instance
        input_size (tuple): Input tensor size (batch_size, channels, depth, height, width)
    """
    print("\n=== MRIEncoder Summary ===")
    summary(
        encoder,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=4,
        device="cpu"
    )

def get_trainable_parameters(encoder):
    """
    Get number of trainable and frozen parameters
    
    Args:
        encoder (MRIEncoder): MRIEncoder instance
        
    Returns:
        tuple: (trainable_params, frozen_params)
    """
    trainable_params = 0
    frozen_params = 0
    
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
            
    return trainable_params, frozen_params

def unfreeze_last_n_layers(encoder, n_layers=1):
    """
    Unfreeze the last n layers of the ResNet backbone
    
    Args:
        encoder (MRIEncoder): MRIEncoder instance
        n_layers (int): Number of layers to unfreeze from the end
        Options:
        - 1: Unfreeze only the last batch norm layer (layer4.2.bn3)
        - 2: Unfreeze the last bottleneck block (layer4.2)
        - 3: Unfreeze the last two bottleneck blocks (layer4.1, layer4.2)
        - 4: Unfreeze all of layer4
        
    Returns:
        MRIEncoder: Updated encoder with unfrozen layers
    """
    # Get all layers in the backbone
    layers = []
    for name, module in encoder.encoder.named_modules():
        if isinstance(module, (nn.Conv3d, nn.BatchNorm3d, nn.Linear)):
            layers.append((name, module))
    
    # Group layers by bottleneck blocks
    bottleneck_blocks = {}
    for name, module in layers:
        if 'layer4' in name:
            block_num = name.split('.')[1]  # Get block number (0, 1, or 2)
            if block_num not in bottleneck_blocks:
                bottleneck_blocks[block_num] = []
            bottleneck_blocks[block_num].append((name, module))
    
    # Sort blocks by number
    sorted_blocks = sorted(bottleneck_blocks.items(), key=lambda x: int(x[0]))
    
    # Unfreeze based on n_layers
    if n_layers == 1:
        # Unfreeze only the last batch norm layer
        last_bn = sorted_blocks[-1][1][-1]  # Get last batch norm layer
        for param in last_bn[1].parameters():
            param.requires_grad = True
        print(f"Unfroze layer: {last_bn[0]}")
    
    elif n_layers == 2:
        # Unfreeze the last bottleneck block
        last_block = sorted_blocks[-1][1]
        for name, module in last_block:
            for param in module.parameters():
                param.requires_grad = True
            print(f"Unfroze layer: {name}")
    
    elif n_layers == 3:
        # Unfreeze the last two bottleneck blocks
        for block in sorted_blocks[-2:]:
            for name, module in block[1]:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfroze layer: {name}")
    
    elif n_layers == 4:
        # Unfreeze all of layer4
        for block in sorted_blocks:
            for name, module in block[1]:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfroze layer: {name}")
    
    return encoder

def unfreeze_specific_layers(encoder, layer_names):
    """
    Unfreeze specific layers by their names
    
    Args:
        encoder (MRIEncoder): MRIEncoder instance
        layer_names (list): List of layer names to unfreeze
        
    Returns:
        MRIEncoder: Updated encoder with unfrozen layers
    """
    for name, module in encoder.encoder.named_modules():
        if name in layer_names:
            for param in module.parameters():
                param.requires_grad = True
            print(f"Unfroze layer: {name}")
    
    return encoder

def plot_parameter_distribution(encoder):
    """
    Plot distribution of parameter values in the encoder
    
    Args:
        encoder (MRIEncoder): MRIEncoder instance
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Parameter value distribution
    plt.subplot(1, 2, 1)
    all_params = []
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            all_params.extend(param.detach().cpu().numpy().flatten())
    
    plt.hist(all_params, bins=50)
    plt.title('Distribution of Parameter Values')
    plt.xlabel('Parameter Value')
    plt.ylabel('Count')
    
    # Plot 2: Layer-wise parameter count
    plt.subplot(1, 2, 2)
    layer_params = OrderedDict()
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            layer_name = name.split('.')[0]
            if layer_name not in layer_params:
                layer_params[layer_name] = 0
            layer_params[layer_name] += param.numel()
    
    plt.bar(layer_params.keys(), layer_params.values())
    plt.title('Number of Parameters per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/encoder_parameter_analysis.png')
    plt.close()

def main():
    # Create encoder instance
    encoder = MRIEncoder(model_name="resnet50", pretrained=True, freeze=True)
    
    # Print architecture
    print_encoder_architecture(encoder)
    
    # Print detailed summary
    print_encoder_summary(encoder)
    
    # Get parameter counts
    trainable, frozen = get_trainable_parameters(encoder)
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")
    
    # Unfreeze last layer
    print("\nUnfreezing last layer...")
    encoder = unfreeze_last_n_layers(encoder, n_layers=1)
    
    # Get updated parameter counts
    trainable, frozen = get_trainable_parameters(encoder)
    print(f"\nAfter unfreezing:")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")
    
    # Plot parameter distribution
    plot_parameter_distribution(encoder)

if __name__ == "__main__":
    main() 