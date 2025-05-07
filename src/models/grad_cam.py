import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: str = "cuda"
    ):
        """
        Grad-CAM implementation for 3D MRI visualization.
        
        Args:
            model: The trained model
            target_layer: Name of the target layer for Grad-CAM
            device: Device to run the model on
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        # Register hooks
        self.activations = None
        self.gradients = None
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Get the target layer
        target = model
        for name in target_layer.split('.'):
            target = getattr(target, name)
            
        # Register hooks
        self.forward_handle = target.register_forward_hook(forward_hook)
        self.backward_handle = target.register_backward_hook(backward_hook)
        
    def __del__(self):
        """Remove hooks when object is deleted."""
        self.forward_handle.remove()
        self.backward_handle.remove()
        
    def generate_cam(
        self,
        image: torch.Tensor,
        demographic: torch.Tensor,
        target_score_idx: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific cognitive score.
        
        Args:
            image: MRI image tensor
            demographic: Demographic features tensor
            target_score_idx: Index of the target cognitive score
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(image, demographic)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target score
        target = output[:, target_score_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(2, 3, 4))
        
        # Generate heatmap
        heatmap = torch.zeros(activations.shape[2:])
        for i, w in enumerate(weights[0]):
            heatmap += w * activations[0, i, :, :, :]
            
        # ReLU and normalize
        heatmap = torch.relu(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap.numpy()
    
    def visualize(
        self,
        image: torch.Tensor,
        demographic: torch.Tensor,
        target_score_idx: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize Grad-CAM heatmap overlaid on MRI slices.
        
        Args:
            image: MRI image tensor
            demographic: Demographic features tensor
            target_score_idx: Index of the target cognitive score
            save_path: Optional path to save visualization
        """
        # Generate heatmap
        heatmap = self.generate_cam(image, demographic, target_score_idx)
        
        # Get original image
        image = image[0, 0].detach().cpu().numpy()
        
        # Create figure with ImageGrid
        fig = plt.figure(figsize=(15, 5))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(1, 3),
            axes_pad=0.1,
            share_all=True
        )
        
        # Plot middle slices in each dimension
        middle_slices = [s // 2 for s in image.shape]
        
        for i, (ax, slice_idx) in enumerate(zip(grid, middle_slices)):
            if i == 0:  # Axial
                img_slice = image[slice_idx, :, :]
                heat_slice = heatmap[slice_idx, :, :]
            elif i == 1:  # Coronal
                img_slice = image[:, slice_idx, :]
                heat_slice = heatmap[:, slice_idx, :]
            else:  # Sagittal
                img_slice = image[:, :, slice_idx]
                heat_slice = heatmap[:, :, slice_idx]
                
            # Plot original image
            ax.imshow(img_slice, cmap='gray')
            
            # Overlay heatmap
            ax.imshow(heat_slice, cmap='jet', alpha=0.5)
            ax.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 