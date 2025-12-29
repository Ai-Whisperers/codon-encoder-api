"""
Codon Basin Counter - Skeptical Validation
Counts basin occupancy without forcing genetic code semantics.
Treats the 64 codons as learned attractors in hyperbolic space.
"""
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional


class CodonEncoder(nn.Module):
    """Minimal encoder architecture matching checkpoint."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


BASES = ['T', 'C', 'A', 'G']
ALL_64_CODONS = [b1 + b2 + b3 for b1 in BASES for b2 in BASES for b3 in BASES]


class CodonBasinCounter:
    """
    Counts codon basin occupancy using learned embeddings.

    No semantic assumptions - just measures which of the 64
    possible codon attractors each input falls into.
    """

    def __init__(self, checkpoint_path: str = None):
        import os
        from pathlib import Path
        if checkpoint_path is None:
            default_model = Path(__file__).parent.parent / "model" / "codon_encoder.pt"
            checkpoint_path = os.getenv('CODON_MODEL_PATH', str(default_model))
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        self.model = CodonEncoder()
        # Load only encoder weights
        encoder_state = {k: v for k, v in self.checkpoint['model_state'].items()
                         if k.startswith('encoder')}
        self.model.load_state_dict(encoder_state)
        self.model.eval()

        # Build reference embeddings for all 64 codons
        self.codon_embeddings = self._build_codon_embeddings()
        self.basin_counts = Counter()

    def _codon_to_onehot(self, codon: str) -> np.ndarray:
        """Convert 3-letter codon to 12-dim one-hot."""
        base_idx = {'T': 0, 'C': 1, 'A': 2, 'G': 3}
        onehot = np.zeros(12, dtype=np.float32)
        for i, base in enumerate(codon.upper()):
            if base in base_idx:
                onehot[i * 4 + base_idx[base]] = 1.0
        return onehot

    def _build_codon_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute reference embedding for each of 64 codons."""
        embeddings = {}
        for codon in ALL_64_CODONS:
            onehot = torch.tensor(self._codon_to_onehot(codon)).unsqueeze(0)
            with torch.no_grad():
                emb = self.model(onehot).squeeze().numpy()
            embeddings[codon] = emb
        return embeddings

    def find_nearest_basin(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Find which of 64 codon basins this embedding falls into."""
        min_dist = float('inf')
        nearest = None

        for codon, ref_emb in self.codon_embeddings.items():
            dist = np.linalg.norm(embedding - ref_emb)
            if dist < min_dist:
                min_dist = dist
                nearest = codon

        return nearest, min_dist

    def encode_and_count(self, sequence: str) -> Dict[str, int]:
        """
        Encode sequence and count basin occupancy.

        Args:
            sequence: DNA sequence (extracts valid codons)

        Returns:
            Counter of which basins were hit
        """
        # Clean and extract codons
        clean = ''.join(c.upper() for c in sequence if c.upper() in 'ATCG')
        codons = [clean[i:i+3] for i in range(0, len(clean) - 2, 3)]

        session_counts = Counter()

        for codon in codons:
            if len(codon) != 3:
                continue

            onehot = torch.tensor(self._codon_to_onehot(codon)).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(onehot).squeeze().numpy()

            basin, dist = self.find_nearest_basin(embedding)
            session_counts[basin] += 1
            self.basin_counts[basin] += 1

        return dict(session_counts)

    def validate_basin_stability(self) -> Dict[str, dict]:
        """
        Skeptical validation: check if each codon maps to itself.

        Returns dict with:
          - self_mapping: bool (does codon map to its own basin?)
          - nearest_dist: distance to nearest basin
          - second_dist: distance to second-nearest (separation margin)
        """
        results = {}

        for codon in ALL_64_CODONS:
            emb = self.codon_embeddings[codon]

            # Find two nearest basins
            distances = []
            for other, other_emb in self.codon_embeddings.items():
                dist = np.linalg.norm(emb - other_emb)
                distances.append((other, dist))

            distances.sort(key=lambda x: x[1])
            nearest, nearest_dist = distances[0]
            second, second_dist = distances[1]

            results[codon] = {
                'self_mapping': nearest == codon,
                'nearest': nearest,
                'nearest_dist': nearest_dist,
                'second': second,
                'second_dist': second_dist,
                'margin': second_dist - nearest_dist
            }

        return results

    def get_basin_stats(self) -> dict:
        """Return summary statistics of basin occupancy."""
        if not self.basin_counts:
            return {'error': 'No sequences processed yet'}

        counts = list(self.basin_counts.values())
        total = sum(counts)

        return {
            'total_codons': total,
            'unique_basins_hit': len(self.basin_counts),
            'most_common': self.basin_counts.most_common(10),
            'coverage': len(self.basin_counts) / 64,
            'entropy': self._compute_entropy(counts, total)
        }

    def _compute_entropy(self, counts: List[int], total: int) -> float:
        """Shannon entropy of basin distribution."""
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * np.log2(p)
        return entropy

    def reset_counts(self):
        """Reset accumulated basin counts."""
        self.basin_counts = Counter()


def main():
    print("Initializing Codon Basin Counter...")
    counter = CodonBasinCounter()

    # Validate basin stability
    print("\n" + "=" * 50)
    print("BASIN STABILITY VALIDATION")
    print("=" * 50)

    stability = counter.validate_basin_stability()

    self_mapped = sum(1 for v in stability.values() if v['self_mapping'])
    print(f"Self-mapping codons: {self_mapped}/64 ({self_mapped/64:.1%})")

    # Show any codons that don't self-map
    non_self = [(k, v) for k, v in stability.items() if not v['self_mapping']]
    if non_self:
        print(f"\nCodons mapping to different basins:")
        for codon, info in non_self[:10]:
            print(f"  {codon} -> {info['nearest']} (margin: {info['margin']:.4f})")

    # Show margin distribution
    margins = [v['margin'] for v in stability.values()]
    print(f"\nSeparation margins:")
    print(f"  Min: {min(margins):.6f}")
    print(f"  Max: {max(margins):.6f}")
    print(f"  Mean: {np.mean(margins):.6f}")

    # Test with a sequence
    print("\n" + "=" * 50)
    print("BASIN OCCUPANCY TEST")
    print("=" * 50)

    test_seq = "ATGGCTCTGTGGATGCGCCTGCTGCCCCTGCTGGCGCTGCTGGCCCTGTGG"
    counts = counter.encode_and_count(test_seq)

    print(f"Sequence: {test_seq[:30]}...")
    print(f"Codons processed: {sum(counts.values())}")
    print(f"Unique basins hit: {len(counts)}")
    print(f"\nBasin occupancy:")
    for codon, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {codon}: {count}")

    stats = counter.get_basin_stats()
    print(f"\nEntropy: {stats['entropy']:.3f} bits")
    print(f"Max entropy (uniform): {np.log2(len(counts)):.3f} bits")


if __name__ == "__main__":
    main()
