import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple

class InstanceNorm(nn.Module):
    """Instance Normalization for graph data"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        return (x - mean) / std

class GNN_F(nn.Module):
    """
    GNN F_Θ architecture from the paper
    Uses both identity and graph convolution operators: F = {I, Â_k}
    """
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input layer
        self.conv1_1 = nn.Linear(input_dim, hidden_dim)
        self.conv1_2 = GCNConv(input_dim, hidden_dim)
        
        # Hidden layers
        self.convs1 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-2)
        ])
        self.convs2 = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)
        ])
        
        # Output layer
        self.conv_out1 = nn.Linear(hidden_dim, output_dim)
        self.conv_out2 = GCNConv(hidden_dim, output_dim)
        
        # Normalization and dropout
        self.instance_norm = InstanceNorm()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input layer
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x, edge_index)
        x = F.relu(x1 + x2)
        x = self.instance_norm(x)
        x = self.dropout(x)
        
        # Hidden layers
        for i in range(self.num_layers-2):
            x1 = self.convs1[i](x)
            x2 = self.convs2[i](x, edge_index)
            x = F.relu(x1 + x2)
            x = self.instance_norm(x)
            x = self.dropout(x)
            
        # Output layer - save penultimate features
        penultimate = x
        x1 = self.conv_out1(x)
        x2 = self.conv_out2(x, edge_index)
        x = x1 + x2
        
        return x, penultimate

class GNN_F_Prime(nn.Module):
    """
    GNN F'_Θ architecture from the paper
    Uses only graph convolution operator: F' = {Â_k}
    """
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Hidden layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)
        ])
        
        # Output layer
        self.conv_out = GCNConv(hidden_dim, output_dim)
        
        # Normalization and dropout
        self.instance_norm = InstanceNorm()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.instance_norm(x)
        x = self.dropout(x)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.instance_norm(x)
            x = self.dropout(x)
            
        # Output layer - save penultimate features
        penultimate = x
        x = self.conv_out(x, edge_index)
        
        return x, penultimate