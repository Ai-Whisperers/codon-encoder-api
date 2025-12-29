"""
Codon Encoder Inference - Hyperbolic Space
Encodes DNA codons into a learned hyperbolic embedding space.
"""
import torch
import torch.nn as nn
import numpy as np

# Standard codon table
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

BASE_TO_IDX = {'T': 0, 'C': 1, 'A': 2, 'G': 3}


class CodonEncoder(nn.Module):
    """Codon encoder with cluster head."""

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

    def encode(self, x):
        return self.encoder(x)


def codon_to_onehot(codon):
    """Convert codon to 12-dim one-hot (4 bases × 3 positions)."""
    onehot = np.zeros(12, dtype=np.float32)
    for i, base in enumerate(codon.upper()):
        if base in BASE_TO_IDX:
            onehot[i * 4 + BASE_TO_IDX[base]] = 1.0
    return onehot


def extract_codons(dna_sequence):
    """Extract codons from DNA sequence (reading frame 0)."""
    # Clean sequence
    seq = ''.join(c.upper() for c in dna_sequence if c.upper() in 'ATCG')
    codons = []
    for i in range(0, len(seq) - 2, 3):
        codons.append(seq[i:i+3])
    return codons


def compute_hyperbolic_radius(embedding):
    """Compute radius in Poincaré ball model."""
    norm = np.linalg.norm(embedding)
    return norm  # In Poincaré ball, radius is just the norm (clamped to <1)


# Hierarchical branching factor (internal constant)
_HIERARCHY_FACTOR = 3

def compute_depth_level(position, max_depth=9):
    """Compute hierarchical depth from position index."""
    if position == 0:
        return max_depth
    d = 0
    p = position
    while p % _HIERARCHY_FACTOR == 0 and d < max_depth:
        p //= _HIERARCHY_FACTOR
        d += 1
    return d


def find_nearest_cluster(embedding, cluster_centers):
    """Find nearest cluster using hyperbolic distance (simplified as Euclidean for Poincare ball)."""
    distances = np.linalg.norm(cluster_centers - embedding, axis=1)
    return np.argmin(distances), distances


def build_empirical_cluster_mapping(model, cluster_centers):
    """Build cluster->AA mapping by testing all 64 codons."""
    cluster_to_aa = {}

    for codon, aa in CODON_TABLE.items():
        onehot = torch.tensor(codon_to_onehot(codon)).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode(onehot).squeeze().numpy()

        cluster_idx, _ = find_nearest_cluster(embedding, cluster_centers)

        if cluster_idx not in cluster_to_aa:
            cluster_to_aa[cluster_idx] = aa

    return cluster_to_aa


def main():
    # Load checkpoint
    import os
    from pathlib import Path
    default_model = Path(__file__).parent / "model" / "codon_encoder.pt"
    model_path = os.getenv('CODON_MODEL_PATH', str(default_model))
    print(f"Loading codon encoder from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Initialize model
    model = CodonEncoder()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    codon_to_position = checkpoint['codon_to_position']
    aa_to_cluster = checkpoint['aa_to_cluster']
    metadata = checkpoint['metadata']
    cluster_centers = checkpoint['model_state']['cluster_centers'].numpy()

    # Build empirical mapping using nearest-neighbor to cluster centers
    cluster_idx_to_aa = build_empirical_cluster_mapping(model, cluster_centers)

    print(f"\nModel version: {metadata['version']}")
    print(f"Structure score: {metadata.get('hierarchy_correlation', 0):.4f}")
    print(f"Cluster accuracy: {metadata['cluster_accuracy']:.2%}")
    print(f"Synonymous accuracy: {metadata['synonymous_accuracy']:.2%}")

    # Example: Human insulin signal peptide (first 24 codons)
    # MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKR
    # DNA: ATG GCT CTG TGG ATG CGC CTG CTG CCC CTG CTG GCG CTG CTG GCC CTG TGG...

    dna_sequence = """
    ATGGCTCTGTGGATGCGCCTGCTGCCCCTGCTGGCGCTGCTGGCCCTGTGGGCGCCCGACCCAGCCGCGGCC
    """

    print(f"\n{'='*60}")
    print("INPUT DNA SEQUENCE:")
    print(f"{'='*60}")
    clean_seq = ''.join(c.upper() for c in dna_sequence if c.upper() in 'ATCG')
    print(clean_seq[:60] + "..." if len(clean_seq) > 60 else clean_seq)

    codons = extract_codons(dna_sequence)
    print(f"\nExtracted {len(codons)} codons")

    print(f"\n{'='*60}")
    print("CODON ANALYSIS (Hyperbolic Embeddings)")
    print(f"{'='*60}")
    print(f"{'Codon':<6} {'AA':<3} {'Pred':<5} {'Position':<10} {'Depth':<6} {'Radius':<8} {'Cluster Prob'}")
    print("-" * 70)

    embeddings = []
    for codon in codons:
        if codon not in CODON_TABLE:
            continue

        true_aa = CODON_TABLE[codon]

        # Encode
        onehot = torch.tensor(codon_to_onehot(codon)).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode(onehot)

        embedding_np = embedding.squeeze().numpy()
        embeddings.append(embedding_np)

        # Predictions using nearest-neighbor to cluster centers
        pred_cluster, distances = find_nearest_cluster(embedding_np, cluster_centers)
        pred_aa = cluster_idx_to_aa.get(pred_cluster, '?')
        min_dist = distances[pred_cluster]
        confidence = 1.0 / (1.0 + min_dist)  # Convert distance to confidence

        # Radial properties
        position = codon_to_position.get(codon, 0)
        depth = compute_depth_level(position)
        radius = compute_hyperbolic_radius(embedding_np)

        match = "+" if pred_aa == true_aa else "-"
        print(f"{codon:<6} {true_aa:<3} {pred_aa}{match:<4} {position:<10} d={depth:<4} r={radius:.4f}  dist={min_dist:.3f}")

    embeddings = np.array(embeddings)

    print(f"\n{'='*60}")
    print("PROTEIN TRANSLATION")
    print(f"{'='*60}")
    protein = ''.join(CODON_TABLE.get(c, 'X') for c in codons)
    print(protein)

    print(f"\n{'='*60}")
    print("EMBEDDING STATISTICS")
    print(f"{'='*60}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Mean radius: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"Std radius: {np.linalg.norm(embeddings, axis=1).std():.4f}")
    print(f"Min radius: {np.linalg.norm(embeddings, axis=1).min():.4f}")
    print(f"Max radius: {np.linalg.norm(embeddings, axis=1).max():.4f}")

    # Cluster centers analysis
    centers = checkpoint['model_state']['cluster_centers'].numpy()
    print(f"\nCluster center radii (amino acid structure):")
    for aa, idx in sorted(aa_to_cluster.items(), key=lambda x: np.linalg.norm(centers[x[1]])):
        r = np.linalg.norm(centers[idx])
        print(f"  {aa}: r={r:.4f}")


def encode_custom_sequence(dna_input):
    """Encode a custom DNA sequence."""
    import os
    from pathlib import Path
    default_model = Path(__file__).parent / "model" / "codon_encoder.pt"
    model_path = os.getenv('CODON_MODEL_PATH', str(default_model))
    print(f"Loading codon encoder from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = CodonEncoder()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    cluster_centers = checkpoint['model_state']['cluster_centers'].numpy()
    codon_to_position = checkpoint['codon_to_position']
    cluster_idx_to_aa = build_empirical_cluster_mapping(model, cluster_centers)

    codons = extract_codons(dna_input)

    print(f"\nInput: {dna_input[:50]}..." if len(dna_input) > 50 else f"\nInput: {dna_input}")
    print(f"Extracted {len(codons)} codons\n")

    results = []
    for codon in codons:
        if codon not in CODON_TABLE:
            continue

        true_aa = CODON_TABLE[codon]
        onehot = torch.tensor(codon_to_onehot(codon)).unsqueeze(0)

        with torch.no_grad():
            embedding = model.encode(onehot).squeeze().numpy()

        pred_cluster, distances = find_nearest_cluster(embedding, cluster_centers)
        pred_aa = cluster_idx_to_aa.get(pred_cluster, '?')
        position = codon_to_position.get(codon, 0)
        depth = compute_depth_level(position)
        radius = compute_hyperbolic_radius(embedding)

        results.append({
            'codon': codon,
            'true_aa': true_aa,
            'pred_aa': pred_aa,
            'position': position,
            'depth': depth,
            'radius': radius,
            'embedding': embedding
        })

    # Print results
    print("Codon  AA  Radius  Depth  Position")
    print("-" * 40)
    for r in results:
        print(f"{r['codon']}    {r['true_aa']}   {r['radius']:.4f}  d={r['depth']}    {r['position']}")

    protein = ''.join(r['true_aa'] for r in results)
    print(f"\nProtein: {protein}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Custom sequence from command line
        dna = ' '.join(sys.argv[1:])
        encode_custom_sequence(dna)
    else:
        main()
