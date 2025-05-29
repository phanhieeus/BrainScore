import torch
import torch.nn as nn


class SelfInteraction(nn.Module):
    def __init__(self, input_dim):
        """
        Self-interaction module between attributes of a feature vector
        
        Args:
            input_dim (int): Dimension of input vector
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Linear layer to project back to input dimension
        self.projection = nn.Linear(
            input_dim + input_dim * (input_dim - 1) // 2,
            input_dim
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor shape (batch_size, input_dim)
        """
        batch_size = x.size(0)
        
        # Create interaction matrix
        interactions = []
        for i in range(self.input_dim):
            for j in range(i + 1, self.input_dim):
                interaction = x[:, i] * x[:, j]
                interactions.append(interaction)
        
        # Stack interactions
        if interactions:
            interactions = torch.stack(interactions, dim=1)
            # Concatenate with original vector
            combined = torch.cat([x, interactions], dim=1)
            # Project back to input dimension
            return self.projection(combined)
        return x
    
    def get_output_dim(self):
        """Return dimension of output vector"""
        return self.input_dim


class CrossInteraction(nn.Module):
    def __init__(self, dim1, dim2):
        """
        Cross-interaction module between two feature vectors
        
        Args:
            dim1 (int): Dimension of first vector
            dim2 (int): Dimension of second vector
        """
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x1, x2):
        """
        Forward pass
        
        Args:
            x1 (torch.Tensor): Input tensor 1 shape (batch_size, dim1)
            x2 (torch.Tensor): Input tensor 2 shape (batch_size, dim2)
            
        Returns:
            torch.Tensor: Output tensor shape (batch_size, dim1 + dim2)
        """
        batch_size = x1.size(0)
        
        # Create interaction matrix
        x1_expanded = x1.unsqueeze(2)  # (batch_size, dim1, 1)
        x2_expanded = x2.unsqueeze(1)  # (batch_size, 1, dim2)
        
        # Calculate outer product
        interaction = x1_expanded * x2_expanded  # (batch_size, dim1, dim2)
        
        # Calculate mean along each dimension
        mean_dim1 = interaction.mean(dim=2)  # (batch_size, dim1)
        mean_dim2 = interaction.mean(dim=1)  # (batch_size, dim2)
        
        # Concatenate results
        return torch.cat([mean_dim1, mean_dim2], dim=1)
    
    def get_output_dim(self):
        """Return dimension of output vector"""
        return self.dim1 + self.dim2 