import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from torch_geometric.loader import DataLoader

from dataset import create_sbm_dataset
from model import GNN_F, GNN_F_Prime

class NCMetrics:
    """Helper class to compute Neural Collapse metrics"""
    @staticmethod
    def compute_class_means(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute class means and global mean"""
        n = features.size(0) // num_classes
        class_means = []
        for c in range(num_classes):
            class_mask = labels == c
            class_mean = features[class_mask].mean(dim=0)
            class_means.append(class_mean)
        class_means = torch.stack(class_means)
        global_mean = features.mean(dim=0)
        return class_means, global_mean

    @staticmethod
    def compute_covariances(features: torch.Tensor, labels: torch.Tensor, 
                          class_means: torch.Tensor, global_mean: torch.Tensor, 
                          num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute within-class and between-class covariances"""
        n = features.size(0) // num_classes
        
        # Within-class covariance
        sigma_W = torch.zeros_like(features[0].unsqueeze(1) @ features[0].unsqueeze(0))
        for c in range(num_classes):
            class_mask = labels == c
            centered = features[class_mask] - class_means[c]
            sigma_W += (centered.T @ centered) / (num_classes * n)
            
        # Between-class covariance
        sigma_B = torch.zeros_like(features[0].unsqueeze(1) @ features[0].unsqueeze(0))
        for c in range(num_classes):
            centered = class_means[c] - global_mean
            sigma_B += (centered.unsqueeze(1) @ centered.unsqueeze(0)) / num_classes
            
        return sigma_W, sigma_B

    @staticmethod
    def compute_NC1_metrics(sigma_W: torch.Tensor, sigma_B: torch.Tensor) -> Tuple[float, float]:
        """Compute NC1 and NC1-tilde metrics"""
        # Handle numerical stability
        eps = 1e-8
        sigma_B_pinv = torch.pinverse(sigma_B + eps * torch.eye(sigma_B.size(0), device=sigma_B.device))
        
        NC1 = torch.trace(sigma_W @ sigma_B_pinv).item()
        NC1_tilde = (torch.trace(sigma_W) / (torch.trace(sigma_B) + eps)).item()
        
        return NC1, NC1_tilde

def compute_overlap(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Compute overlap score between predictions and ground truth"""
    n = labels.size(0)
    
    # Get predictions
    _, preds = outputs.max(dim=1)
    
    # Try all permutations for 2 classes
    if num_classes == 2:
        orig_acc = (preds == labels).float().mean()
        perm_acc = (preds == (1 - labels)).float().mean()
        accuracy = max(orig_acc, perm_acc)
    else:
        # For more classes, use just original order for simplicity
        accuracy = (preds == labels).float().mean()
        
    # Convert to overlap score
    overlap = (accuracy.item() - 1/num_classes) / (1 - 1/num_classes)
    return overlap

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_overlap = 0
    num_batches = 0
    
    nc1_vals = []
    nc1_tilde_vals = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs, penultimate = model(data.x, data.edge_index)
        
        # Compute MSE loss
        loss = criterion(outputs, F.one_hot(data.y, num_classes=num_classes).float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        overlap = compute_overlap(outputs, data.y, num_classes)
        
        # Compute NC metrics on penultimate features
        class_means, global_mean = NCMetrics.compute_class_means(penultimate, data.y, num_classes)
        sigma_W, sigma_B = NCMetrics.compute_covariances(penultimate, data.y, class_means, global_mean, num_classes)
        nc1, nc1_tilde = NCMetrics.compute_NC1_metrics(sigma_W, sigma_B)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_overlap += overlap
        nc1_vals.append(nc1)
        nc1_tilde_vals.append(nc1_tilde)
        num_batches += 1
        
    # Average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'overlap': total_overlap / num_batches,
        'nc1': np.mean(nc1_vals),
        'nc1_tilde': np.mean(nc1_tilde_vals)
    }
    
    return metrics

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    total_loss = 0
    total_overlap = 0
    num_batches = 0
    
    nc1_vals = []
    nc1_tilde_vals = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            outputs, penultimate = model(data.x, data.edge_index)
            
            # Compute MSE loss
            loss = criterion(outputs, F.one_hot(data.y, num_classes=num_classes).float())
            
            # Compute metrics
            overlap = compute_overlap(outputs, data.y, num_classes)
            
            # Compute NC metrics
            class_means, global_mean = NCMetrics.compute_class_means(penultimate, data.y, num_classes)
            sigma_W, sigma_B = NCMetrics.compute_covariances(penultimate, data.y, class_means, global_mean, num_classes)
            nc1, nc1_tilde = NCMetrics.compute_NC1_metrics(sigma_W, sigma_B)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_overlap += overlap
            nc1_vals.append(nc1)
            nc1_tilde_vals.append(nc1_tilde)
            num_batches += 1
    
    # Average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'overlap': total_overlap / num_batches,
        'nc1': np.mean(nc1_vals),
        'nc1_tilde': np.mean(nc1_tilde_vals)
    }
    
    return metrics

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Training settings
    num_epochs = 8
    batch_size = 32
    learning_rate = 0.004
    momentum = 0.9
    weight_decay = 5e-4
    
    # Model settings
    num_layers = 32
    hidden_dim = 8
    num_classes = 2
    
    # Create datasets
    train_dataset = create_sbm_dataset(
        num_graphs=1000,
        num_nodes=1000,
        num_communities=num_classes,
        p=0.025,
        q=0.0017,
        feature_dim=hidden_dim
    )
    
    test_dataset = create_sbm_dataset(
        num_graphs=100,
        num_nodes=1000,
        num_communities=num_classes,
        p=0.025,
        q=0.0017,
        feature_dim=hidden_dim
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    models = {
        'GNN_F': GNN_F(
            num_layers=num_layers,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes
        ),
        'GNN_F_Prime': GNN_F_Prime(
            num_layers=num_layers,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes
        )
    }
    
    # Training loop for each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, num_classes)
            
            # Validate
            test_metrics = validate(model, test_loader, criterion, device, num_classes)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Overlap: {train_metrics['overlap']:.4f}, "
                  f"NC1: {train_metrics['nc1']:.4f}, NC1-tilde: {train_metrics['nc1_tilde']:.4f}")
            print(f"Test  - Loss: {test_metrics['loss']:.4f}, Overlap: {test_metrics['overlap']:.4f}, "
                  f"NC1: {test_metrics['nc1']:.4f}, NC1-tilde: {test_metrics['nc1_tilde']:.4f}")
            
        # Save final model
        torch.save(model.state_dict(), f"{model_name}_final.pt")

if __name__ == "__main__":
    main()