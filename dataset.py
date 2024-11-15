import numpy as np
import torch
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.transforms import NormalizeFeatures

def create_sbm_dataset(
    num_graphs: int,
    num_nodes: int,
    num_communities: int,
    p: float,
    q: float,
    feature_dim: int = 8,
    seed: int = None
):
    """
    Create SBM dataset using PyG's built-in functionality
    
    Args:
        num_graphs: Number of graphs to generate
        num_nodes: Total number of nodes per graph
        num_communities: Number of communities/classes
        p: Intra-community edge probability
        q: Inter-community edge probability
        feature_dim: Dimension of node features
        seed: Random seed for reproducibility
    """
    if seed is not None:
        torch.manual_seed(seed)

    a = (p * num_nodes) / np.log(num_nodes)
    b = (q * num_nodes) / np.log(num_nodes)

    assert a >= 0 and b >= 0, "Parameters p,q do not satisfy the conditions"
    
    # # Verify exact recovery condition
    # assert abs(np.sqrt(a) - np.sqrt(b)) > np.sqrt(num_communities), \
    #     "Parameters p,q do not satisfy exact recovery condition"
    
    # Create edge probabilities matrix
    block_sizes = [num_nodes // num_communities] * num_communities
    edge_probs = q * torch.ones(num_communities, num_communities)
    edge_probs.fill_diagonal_(p)
    
    # Generate dataset
    dataset = StochasticBlockModelDataset(
        root='SBM/',  # Temporary directory to store the dataset
        block_sizes=block_sizes,
        edge_probs=edge_probs,
        num_channels=feature_dim,
        num_graphs=num_graphs,
        transform=NormalizeFeatures()
    )
    
    return dataset

# Example usage for D1 dataset from paper
if __name__ == "__main__":
    # Dataset parameters for D1
    train_dataset = create_sbm_dataset(
        num_graphs=1000,
        num_nodes=1000,
        num_communities=2,
        p=0.025,
        q=0.0017,
        feature_dim=8,
        seed=42
    )
    
    # Dataset parameters for test set
    test_dataset = create_sbm_dataset(
        num_graphs=100,
        num_nodes=1000,
        num_communities=2,
        p=0.025,
        q=0.0017,
        feature_dim=8,
        seed=43
    )
    
    # Print dataset info
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Look at first graph
    data = train_dataset[0]
    print("\nFirst graph properties:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Feature matrix shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Labels shape: {data.y.shape}")