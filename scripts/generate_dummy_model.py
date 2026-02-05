#!/usr/bin/env python3
"""
Generate a dummy model file for testing and development.
This creates a mock model that has the same structure as the real model
but with random weights for testing API functionality.
"""

import sys
import random
from pathlib import Path

# Try importing torch, fallback to creating a minimal mock if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Creating minimal mock model file.")


def create_torch_dummy_model(output_path: Path) -> None:
    """Create a dummy model using PyTorch."""
    
    # Mock model architecture
    class DummyCodonEncoder(nn.Module):
        def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create model instance
    model = DummyCodonEncoder()
    
    # Create cluster centers for 21 amino acids (20 + stop)
    cluster_centers = torch.randn(21, 16)  # 16-dim embeddings
    
    # Create dummy model state with same structure as real model
    model_state = {
        'encoder': model.state_dict(),
        'cluster_centers': cluster_centers,
        'embed_dim': 16,
        'num_clusters': 21,
        'version': 'dummy_v1.0.0',
        'created_by': 'generate_dummy_model.py'
    }
    
    # Save the model
    torch.save(model_state, output_path)
    print(f"✓ Created PyTorch dummy model at {output_path}")
    print(f"  Embed dim: {model_state['embed_dim']}")
    print(f"  Clusters: {model_state['num_clusters']}")


def create_mock_model(output_path: Path) -> None:
    """Create a minimal mock model without PyTorch dependencies."""
    
    # Create a simple pickle-like structure
    import pickle
    
    # Generate random embeddings for 21 clusters
    random.seed(42)  # Reproducible
    cluster_centers = [[random.gauss(0, 1) for _ in range(16)] for _ in range(21)]
    
    model_state = {
        'type': 'mock_model',
        'cluster_centers': cluster_centers,
        'embed_dim': 16,
        'num_clusters': 21,
        'version': 'mock_v1.0.0',
        'created_by': 'generate_dummy_model.py',
        'note': 'This is a mock model for testing. Install PyTorch for full functionality.'
    }
    
    # Save as pickle (will fail to load with torch.load but better than nothing)
    with open(output_path, 'wb') as f:
        pickle.dump(model_state, f)
    
    print(f"✓ Created mock model at {output_path}")
    print("  Note: This is a mock model. Install PyTorch for full functionality.")


def main():
    """Main function to generate dummy model."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "server" / "model"
    model_path = model_dir / "codon_encoder.pt"
    
    print("=" * 50)
    print("Dummy Model Generator")
    print("=" * 50)
    print(f"Target path: {model_path}")
    print()
    
    # Check if model already exists
    if model_path.exists():
        response = input(f"Model file already exists at {model_path}. Overwrite? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return 0
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate appropriate model
    if TORCH_AVAILABLE:
        create_torch_dummy_model(model_path)
    else:
        create_mock_model(model_path)
    
    print()
    print("Usage:")
    print("  # Start the API with dummy model")
    print("  cd visualizer")
    print("  python run.py")
    print()
    print("  # Test with health check")
    print("  python scripts/health_check.py --test-encode")
    print()
    print("Note: This dummy model will not produce scientifically meaningful results.")
    print("Contact api@ai-whisperers.org for access to the trained model.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())