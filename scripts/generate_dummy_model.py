import torch
import torch.nn as nn
import os
from pathlib import Path

# Define the model architecture (must match what's in model_loader.py)
class CodonEncoder(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.cluster_head = nn.Linear(embed_dim, n_clusters)
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, embed_dim))

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.cluster_head(embedding)
        return embedding, logits

def main():
    print("Generating dummy Codon Encoder model...")
    
    # 1. ensure directory exists
    model_dir = Path("server/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "codon_encoder.pt"

    # 2. visualizer/config.py mappings
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    # 3. Create dummy data structures
    model = CodonEncoder()
    
    # Randomly initialize weights (standard init is fine)
    
    # Create fake mappings
    codon_to_position = {c: i for i, c in enumerate(CODON_TABLE.keys())}
    
    # Create valid AA to cluster mapping (0-20 for 21 clusters)
    unique_aas = sorted(list(set(CODON_TABLE.values()))) # 21 AAs including stop
    aa_to_cluster = {aa: i for i, aa in enumerate(unique_aas)}
    
    # 4. Construct checkpoint dictionary
    checkpoint = {
        'model_state': model.state_dict(),
        'codon_to_position': codon_to_position,
        'aa_to_cluster': aa_to_cluster,
        'metadata': {
            'version': '0.0.0-dummy',
            'hierarchy_correlation': 0.123,
            'cluster_accuracy': 0.99,
            'synonymous_accuracy': 0.99
        }
    }
    
    # 5. Save
    torch.save(checkpoint, model_path)
    print(f"Saved dummy model to {model_path}")

if __name__ == "__main__":
    main()
