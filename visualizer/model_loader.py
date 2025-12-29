"""
Model loader for Codon Encoder.
Extracts embeddings, computes projections, and prepares visualization data.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path

from config import (
    Config, VIS_CONFIG, ALL_64_CODONS,
    CODON_TABLE, BASE_TO_IDX
)


class CodonEncoder(nn.Module):
    """Codon encoder matching checkpoint architecture."""

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


@dataclass
class CodonPoint:
    """Single codon visualization point."""
    codon: str
    amino_acid: str
    position: int
    depth: int
    embedding: List[float]      # Original 16-dim
    projection: List[float]     # 3D projection
    embedding_norm: float       # Euclidean norm in 16-dim space
    poincare_radius: float      # Radius in Poincare disk (d=0 -> 0.9, d=9 -> 0.1)
    cluster_idx: int
    confidence: float
    margin: float               # Separation from second-nearest (basin boundary)


@dataclass
class VisualizationData:
    """Complete visualization payload."""
    points: List[Dict] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)
    cluster_centers_3d: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)


def codon_to_onehot(codon: str) -> np.ndarray:
    """Convert codon to 12-dim one-hot."""
    onehot = np.zeros(12, dtype=np.float32)
    for i, base in enumerate(codon.upper()):
        if base in BASE_TO_IDX:
            onehot[i * 4 + BASE_TO_IDX[base]] = 1.0
    return onehot


# Hierarchical branching factor (internal constant)
_HIERARCHY_FACTOR = 3

def compute_depth_level(position: int, max_depth: int = 9) -> int:
    """Compute hierarchical depth from position index."""
    if position == 0:
        return max_depth
    d = 0
    p = position
    while p % _HIERARCHY_FACTOR == 0 and d < max_depth:
        p //= _HIERARCHY_FACTOR
        d += 1
    return d


def depth_to_poincare_radius(depth: int) -> float:
    """Convert depth to Poincare disk radius.

    d=0 (outer, most common) -> r=0.9
    d=9 (center, rarest)     -> r=0.1
    """
    return 0.9 - (depth / 9) * 0.8


class ModelLoader:
    """Load model and extract visualization data."""

    def __init__(self, model_path: Optional[Path] = None):
        # Use provided path or get from Config
        self.model_path = Path(model_path) if model_path else Config.get_model_path()
        self.checkpoint = None
        self.model = None
        self.codon_embeddings: Dict[str, np.ndarray] = {}
        self.codon_points: List[CodonPoint] = []
        self.cluster_centers: Optional[np.ndarray] = None
        self.projections: Optional[np.ndarray] = None  # Store projections
        self.projection_transform = None  # Store transform for reuse
        self._initialized = False

    def load(self) -> 'ModelLoader':
        """Load checkpoint and initialize model."""
        print(f"Loading model from: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.checkpoint = torch.load(
            self.model_path,
            map_location='cpu',
            weights_only=False
        )

        self.model = CodonEncoder()
        self.model.load_state_dict(self.checkpoint['model_state'])
        self.model.eval()

        self.cluster_centers = self.checkpoint['model_state']['cluster_centers'].numpy()
        self.codon_to_position = self.checkpoint.get('codon_to_position', {})
        self.aa_to_cluster = self.checkpoint.get('aa_to_cluster', {})
        self.metadata = self.checkpoint.get('metadata', {})

        # Build cluster -> AA mapping
        self.cluster_to_aa = {v: k for k, v in self.aa_to_cluster.items()}

        return self

    def _ensure_initialized(self):
        """Ensure embeddings and projections are computed."""
        if not self._initialized:
            self.extract_embeddings()
            self.projections = self.compute_projection(
                VIS_CONFIG.get("projection_method", "pca")
            )
            self.build_codon_points(self.projections)
            self._initialized = True

    def extract_embeddings(self) -> Dict[str, np.ndarray]:
        """Extract embeddings for all 64 codons."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for codon in ALL_64_CODONS:
            onehot = torch.tensor(codon_to_onehot(codon)).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model.encode(onehot).squeeze().numpy()
            self.codon_embeddings[codon] = embedding
        return self.codon_embeddings

    def compute_projection(self, method: str = "pca") -> np.ndarray:
        """Project 16-dim embeddings to 3D."""
        if not self.codon_embeddings:
            self.extract_embeddings()

        embeddings = np.array([self.codon_embeddings[c] for c in ALL_64_CODONS])
        self.embedding_mean = embeddings.mean(axis=0)

        if method == "pca":
            pca = PCA(n_components=3)
            projections = pca.fit_transform(embeddings)
            self.projection_transform = {
                'method': 'pca',
                'components': pca.components_,
                'mean': pca.mean_,
                'scale': np.abs(projections).max()
            }
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1)
                projections = reducer.fit_transform(embeddings)
                self.projection_transform = {
                    'method': 'umap',
                    'reducer': reducer,
                    'scale': np.abs(projections).max()
                }
            except ImportError:
                print("UMAP not installed, falling back to PCA")
                return self.compute_projection("pca")
        elif method == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=3, perplexity=15, random_state=42)
            projections = tsne.fit_transform(embeddings)
            # t-SNE doesn't have a transform method, store embedding positions
            self.projection_transform = {
                'method': 'tsne',
                'embeddings': embeddings.copy(),
                'projections': projections.copy(),
                'scale': np.abs(projections).max()
            }
        else:
            raise ValueError(f"Unknown projection method: {method}")

        # Normalize to unit sphere for visualization
        scale = self.projection_transform['scale'] + 1e-8
        projections = projections / scale
        self.projections = projections

        return projections

    def project_point(self, embedding: np.ndarray) -> np.ndarray:
        """Project a single embedding to 3D using stored transform."""
        if self.projection_transform is None:
            raise RuntimeError("Projection not computed. Call compute_projection() first.")

        method = self.projection_transform['method']
        scale = self.projection_transform['scale'] + 1e-8

        if method == 'pca':
            centered = embedding - self.projection_transform['mean']
            projected = centered @ self.projection_transform['components'].T
            return projected / scale
        elif method == 'umap':
            projected = self.projection_transform['reducer'].transform(
                embedding.reshape(1, -1)
            )[0]
            return projected / scale
        elif method == 'tsne':
            # For t-SNE, find nearest neighbor in original space
            stored_emb = self.projection_transform['embeddings']
            stored_proj = self.projection_transform['projections']
            distances = np.linalg.norm(stored_emb - embedding, axis=1)
            nearest_idx = np.argmin(distances)
            return stored_proj[nearest_idx] / scale
        else:
            raise ValueError(f"Unknown projection method: {method}")

    def find_nearest_cluster(self, embedding: np.ndarray) -> Tuple[int, float, float]:
        """Find nearest cluster, return (idx, min_dist, second_dist)."""
        if self.cluster_centers is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        sorted_idx = np.argsort(distances)
        return sorted_idx[0], distances[sorted_idx[0]], distances[sorted_idx[1]]

    def build_codon_points(self, projections: np.ndarray) -> List[CodonPoint]:
        """Build visualization points for all codons."""
        self.codon_points = []

        for i, codon in enumerate(ALL_64_CODONS):
            embedding = self.codon_embeddings[codon]
            position = self.codon_to_position.get(codon, i)
            depth = compute_depth_level(position)
            embedding_norm = float(np.linalg.norm(embedding))
            poincare_radius = depth_to_poincare_radius(depth)

            cluster_idx, min_dist, second_dist = self.find_nearest_cluster(embedding)
            confidence = 1.0 / (1.0 + min_dist)
            margin = second_dist - min_dist

            point = CodonPoint(
                codon=codon,
                amino_acid=CODON_TABLE.get(codon, '?'),
                position=position,
                depth=depth,
                embedding=embedding.tolist(),
                projection=projections[i].tolist(),
                embedding_norm=embedding_norm,
                poincare_radius=poincare_radius,
                cluster_idx=int(cluster_idx),
                confidence=float(confidence),
                margin=float(margin),
            )
            self.codon_points.append(point)

        return self.codon_points

    def compute_edges(self, mode: str = "hierarchical") -> List[Dict]:
        """Compute edges/fibers between codons."""
        # Ensure points are initialized
        self._ensure_initialized()

        edges = []
        threshold = VIS_CONFIG.get("fiber_threshold", 0.5)

        if mode == "none":
            return edges

        for i, p1 in enumerate(self.codon_points):
            for j, p2 in enumerate(self.codon_points):
                if i >= j:
                    continue

                connect = False
                weight = 0.0

                if mode == "hierarchical":
                    # Connect if same depth level
                    if p1.depth == p2.depth:
                        dist = np.linalg.norm(
                            np.array(p1.embedding) - np.array(p2.embedding)
                        )
                        if dist < threshold:
                            connect = True
                            weight = 1.0 - dist / threshold

                elif mode == "amino_acid":
                    # Connect synonymous codons
                    if p1.amino_acid == p2.amino_acid:
                        connect = True
                        weight = 1.0

                elif mode == "depth":
                    # Connect adjacent depth levels
                    if abs(p1.depth - p2.depth) == 1:
                        connect = True
                        weight = 0.5

                if connect:
                    edges.append({
                        "source": i,
                        "target": j,
                        "weight": weight,
                        "type": mode,
                    })

        return edges

    def project_cluster_centers(self) -> List[Dict]:
        """Project cluster centers to 3D using same transform."""
        self._ensure_initialized()

        if self.projection_transform is None:
            raise RuntimeError("Projection not computed")

        method = self.projection_transform['method']

        if method == 'pca':
            centered = self.cluster_centers - self.projection_transform['mean']
            centers_3d = centered @ self.projection_transform['components'].T
            scale = self.projection_transform['scale'] + 1e-8
            centers_3d = centers_3d / scale
        elif method == 'umap':
            centers_3d = self.projection_transform['reducer'].transform(
                self.cluster_centers
            )
            scale = self.projection_transform['scale'] + 1e-8
            centers_3d = centers_3d / scale
        else:
            # Fallback for t-SNE: average position of codons in each cluster
            centers_3d = []
            for idx in range(len(self.cluster_centers)):
                cluster_points = [
                    self.projections[i] for i, p in enumerate(self.codon_points)
                    if p.cluster_idx == idx
                ]
                if cluster_points:
                    centers_3d.append(np.mean(cluster_points, axis=0))
                else:
                    centers_3d.append(np.zeros(3))
            centers_3d = np.array(centers_3d)

        result = []
        for idx, center in enumerate(centers_3d):
            aa = self.cluster_to_aa.get(idx, '?')
            result.append({
                "cluster_idx": idx,
                "amino_acid": aa,
                "position": center.tolist(),
                "radius": float(np.linalg.norm(self.cluster_centers[idx])),
            })
        return result

    def get_visualization_data(self) -> VisualizationData:
        """Get complete visualization payload."""
        self._ensure_initialized()

        # Compute edges and cluster centers
        edges = self.compute_edges(VIS_CONFIG.get("fiber_mode", "hierarchical"))
        centers = self.project_cluster_centers()

        return VisualizationData(
            points=[asdict(p) for p in self.codon_points],
            edges=edges,
            cluster_centers_3d=centers,
            metadata={
                "model_path": str(self.model_path),
                "version": self.metadata.get("version", "unknown"),
                "structure_score": self.metadata.get("hierarchy_correlation", 0),
                "cluster_accuracy": self.metadata.get("cluster_accuracy", 0),
                "n_codons": len(ALL_64_CODONS),
                "embed_dim": 16,
            },
            config=VIS_CONFIG,
        )

    def encode_sequence(
        self,
        dna_sequence: str,
        stop_at_stop_codon: bool = True,
        include_stop: bool = True
    ) -> List[Dict]:
        """
        Encode a DNA sequence for live inference visualization.

        Args:
            dna_sequence: DNA string (ATCG)
            stop_at_stop_codon: If True, stop translation at first stop codon
            include_stop: If True, include the stop codon in results

        Returns:
            List of encoded codon dicts with positions and embeddings
        """
        self._ensure_initialized()

        # Clean and extract codons
        clean = ''.join(c.upper() for c in dna_sequence if c.upper() in 'ATCG')
        codons = [clean[i:i+3] for i in range(0, len(clean) - 2, 3)]

        results = []
        for seq_pos, codon in enumerate(codons):
            if len(codon) < 3 or codon not in CODON_TABLE:
                continue

            amino_acid = CODON_TABLE[codon]
            is_stop = (amino_acid == '*')

            # Handle stop codon logic
            if is_stop and stop_at_stop_codon:
                if include_stop:
                    # Add stop codon then break
                    results.append(self._encode_single_codon(seq_pos, codon))
                break

            # Get embedding and add to results
            results.append(self._encode_single_codon(seq_pos, codon))

        return results

    def _encode_single_codon(self, seq_pos: int, codon: str) -> Dict:
        """Encode a single codon and return result dict."""
        onehot = torch.tensor(codon_to_onehot(codon)).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model.encode(onehot).squeeze().numpy()

        ref_idx = ALL_64_CODONS.index(codon)
        ref_point = self.codon_points[ref_idx]

        cluster_idx, min_dist, second_dist = self.find_nearest_cluster(embedding)

        return {
            "seq_position": seq_pos,
            "codon": codon,
            "amino_acid": CODON_TABLE[codon],
            "ref_idx": ref_idx,
            "projection": ref_point.projection,
            "depth": ref_point.depth,
            "poincare_radius": ref_point.poincare_radius,
            "embedding_norm": ref_point.embedding_norm,
            "cluster_idx": int(cluster_idx),
            "confidence": float(1.0 / (1.0 + min_dist)),
            "margin": float(second_dist - min_dist),
        }

    def to_json(self) -> str:
        """Export visualization data as JSON."""
        data = self.get_visualization_data()
        return json.dumps(asdict(data), indent=2)

    def compute_angular_variance(self) -> Dict:
        """
        Compute intra-AA vs inter-AA angular variance at each depth level.

        Returns:
            Dict with per-depth stats and overall metrics
        """
        self._ensure_initialized()

        # Group codons by depth
        by_depth = {}
        for i, p in enumerate(self.codon_points):
            d = p.depth
            if d not in by_depth:
                by_depth[d] = []
            # Compute angle in Poincare disk using position-based layout
            # This matches the frontend's projectPoint() formula
            base_angle = (i / 64) * np.pi * 2
            jitter = (p.position or i) * 0.1
            angle = base_angle + jitter
            by_depth[d].append({
                'codon': p.codon,
                'amino_acid': p.amino_acid,
                'angle': angle,
                'idx': i
            })

        results = {}
        for d, codons in by_depth.items():
            # Group by amino acid within this depth
            by_aa = {}
            for c in codons:
                aa = c['amino_acid']
                if aa not in by_aa:
                    by_aa[aa] = []
                by_aa[aa].append(c['angle'])

            # Intra-AA variance: average variance within each AA group
            intra_variances = []
            for aa, angles in by_aa.items():
                if len(angles) > 1:
                    # Circular variance for angles
                    angles_arr = np.array(angles)
                    mean_cos = np.mean(np.cos(angles_arr))
                    mean_sin = np.mean(np.sin(angles_arr))
                    # Circular variance = 1 - R, where R is mean resultant length
                    R = np.sqrt(mean_cos**2 + mean_sin**2)
                    circ_var = 1 - R
                    intra_variances.append(circ_var)

            # Inter-AA variance: variance of AA centroids
            aa_centroids = []
            for aa, angles in by_aa.items():
                mean_angle = np.arctan2(
                    np.mean(np.sin(angles)),
                    np.mean(np.cos(angles))
                )
                aa_centroids.append(mean_angle)

            inter_var = 0
            if len(aa_centroids) > 1:
                centroids_arr = np.array(aa_centroids)
                mean_cos = np.mean(np.cos(centroids_arr))
                mean_sin = np.mean(np.sin(centroids_arr))
                R = np.sqrt(mean_cos**2 + mean_sin**2)
                inter_var = 1 - R

            intra_var = np.mean(intra_variances) if intra_variances else 0

            results[d] = {
                'depth': d,
                'n_codons': len(codons),
                'n_amino_acids': len(by_aa),
                'intra_aa_variance': float(intra_var),
                'inter_aa_variance': float(inter_var),
                'ratio': float(inter_var / (intra_var + 1e-8)),
                'amino_acids': list(by_aa.keys())
            }

        # Overall summary
        all_intra = [r['intra_aa_variance'] for r in results.values()]
        all_inter = [r['inter_aa_variance'] for r in results.values()]

        return {
            'by_depth': results,
            'summary': {
                'mean_intra_variance': float(np.mean(all_intra)),
                'mean_inter_variance': float(np.mean(all_inter)),
                'mean_ratio': float(np.mean(all_inter) / (np.mean(all_intra) + 1e-8))
            }
        }

    def generate_synonymous_variants(self, protein_seq: str, n_variants: int = 3) -> List[Dict]:
        """
        Generate synonymous DNA variants for a protein sequence.

        Args:
            protein_seq: Amino acid sequence (e.g., "MDDII...")
            n_variants: Number of variants to generate

        Returns:
            List of dicts with DNA sequences and their encodings
        """
        self._ensure_initialized()

        # Build reverse codon table: AA -> list of codons
        aa_to_codons = {}
        for codon, aa in CODON_TABLE.items():
            if aa not in aa_to_codons:
                aa_to_codons[aa] = []
            aa_to_codons[aa].append(codon)

        variants = []
        for variant_idx in range(n_variants):
            dna = []
            strategy = 'random' if variant_idx == 0 else (
                'high_depth' if variant_idx == 1 else 'low_depth'
            )

            for aa in protein_seq.upper():
                if aa not in aa_to_codons:
                    continue

                codons = aa_to_codons[aa]
                if strategy == 'random':
                    # Random choice
                    chosen = codons[hash((aa, variant_idx)) % len(codons)]
                elif strategy == 'high_depth':
                    # Prefer high depth (center of disk)
                    best = max(codons, key=lambda c: self.codon_points[
                        ALL_64_CODONS.index(c)
                    ].depth)
                    chosen = best
                else:
                    # Prefer low depth (edge of disk)
                    best = min(codons, key=lambda c: self.codon_points[
                        ALL_64_CODONS.index(c)
                    ].depth)
                    chosen = best

                dna.append(chosen)

            dna_seq = ''.join(dna)
            encoded = self.encode_sequence(dna_seq, stop_at_stop_codon=False)

            # Compute trajectory stats
            depths = [e['depth'] for e in encoded]
            variants.append({
                'variant_idx': variant_idx,
                'strategy': strategy,
                'dna_sequence': dna_seq,
                'protein': protein_seq,
                'encoded': encoded,
                'mean_depth': float(np.mean(depths)) if depths else 0,
                'depth_std': float(np.std(depths)) if depths else 0
            })

        return variants


def validate_translation(dna: str, expected_protein: str) -> bool:
    """
    Validate DNA->protein translation using pythonic one-liner.

    Usage:
        assert validate_translation("ATGGCT", "MA")
    """
    t = CODON_TABLE
    result = ''.join(t[''.join(c)] for c in zip(*[iter(dna.upper())]*3))
    return result == expected_protein


def main():
    """Test model loading."""
    loader = ModelLoader().load()
    data = loader.get_visualization_data()

    print(f"Model: {data.metadata['model_path']}")
    print(f"Version: {data.metadata['version']}")
    print(f"Points: {len(data.points)}")
    print(f"Edges: {len(data.edges)}")
    print(f"Cluster centers: {len(data.cluster_centers_3d)}")

    # Test inference
    test_seq = "ATGGCTCTGTGG"
    encoded = loader.encode_sequence(test_seq)
    print(f"\nInference test ({test_seq}):")
    for e in encoded:
        print(f"  {e['codon']} -> {e['amino_acid']} (d={e['depth']})")

    # Validate translation
    expected = ''.join(e['amino_acid'] for e in encoded)
    assert validate_translation(test_seq, expected), "Translation validation failed!"
    print(f"\nValidation: OK ('{expected}')")


if __name__ == "__main__":
    main()
