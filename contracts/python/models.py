"""
Codon Inference API - Python Client Models

API endpoint: https://codon.ai-whisperers.org/api

These Pydantic models define the public contract for consuming the Codon Encoder
inference service. Model weights and raw embeddings are not exposed.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Core Data Types
# =============================================================================


class CodonPoint(BaseModel):
    """
    Single codon with inference results.

    Note: Raw 16-dim embeddings are NOT exposed. Only derived metrics
    (projection, confidence, margin) are available.
    """

    codon: str = Field(..., description="Three-letter codon (e.g., 'ATG')")
    amino_acid: str = Field(..., description="Single-letter amino acid code")
    position: int = Field(..., ge=0, le=63, description="Position in canonical 64-codon ordering")
    depth: int = Field(..., ge=0, le=9, description="Hierarchical depth level")
    projection: Tuple[float, float, float] = Field(..., description="3D projection coordinates")
    embedding_norm: float = Field(..., ge=0, description="Euclidean norm of internal embedding")
    poincare_radius: float = Field(..., ge=0, le=1, description="Hyperbolic radius in Poincaré disk")
    cluster_idx: int = Field(..., ge=0, le=20, description="Predicted amino acid cluster index")
    confidence: float = Field(..., ge=0, le=1, description="Cluster assignment confidence")
    margin: float = Field(..., description="Decision margin to second-nearest cluster")
    color: str = Field(..., description="Display color (hex string)")

    @field_validator("codon")
    @classmethod
    def validate_codon(cls, v: str) -> str:
        if len(v) != 3 or not all(c in "ATCG" for c in v.upper()):
            raise ValueError("Codon must be 3 characters from ATCG")
        return v.upper()


class Edge(BaseModel):
    """Edge/fiber connecting two codons in visualization."""

    source: int = Field(..., ge=0, le=63, description="Index of source codon")
    target: int = Field(..., ge=0, le=63, description="Index of target codon")
    weight: float = Field(..., ge=0, le=1, description="Connection strength")


class ModelMetadata(BaseModel):
    """Model metadata (public information only)."""

    version: str = Field(..., description="Model version identifier")
    structure_score: float = Field(..., description="Structure quality score (Spearman correlation)")
    cluster_accuracy: float = Field(..., ge=0, le=1, description="Cluster classification accuracy")
    embed_dim: int = Field(..., description="Embedding dimensionality (informational)")
    num_clusters: int = Field(default=21, description="Number of amino acid clusters")
    projection_method: str = Field(default="PCA", description="Projection method used")


# =============================================================================
# API Request Types
# =============================================================================


class EncodeRequest(BaseModel):
    """Request body for sequence encoding."""

    sequence: str = Field(..., min_length=3, description="DNA sequence (ATCG characters)")

    @field_validator("sequence")
    @classmethod
    def clean_sequence(cls, v: str) -> str:
        """Remove non-ATCG characters and uppercase."""
        return "".join(c for c in v.upper() if c in "ATCG")


class SynonymousVariantsRequest(BaseModel):
    """Request body for synonymous variant generation."""

    protein: str = Field(..., min_length=1, description="Protein sequence (single-letter AA codes)")
    n_variants: int = Field(default=3, ge=1, le=10, description="Number of variants to generate")


# =============================================================================
# API Response Types
# =============================================================================


class EncodedCodon(BaseModel):
    """Single codon in an encoded sequence trajectory."""

    seq_position: int = Field(..., ge=0, description="Position in input sequence (0-indexed)")
    codon: str = Field(..., description="Three-letter codon")
    amino_acid: str = Field(..., description="Translated amino acid")
    ref_idx: int = Field(..., ge=0, le=63, description="Reference index in canonical 64-codon list")
    projection: Tuple[float, float, float] = Field(..., description="3D projection coordinates")
    depth: int = Field(..., ge=0, le=9, description="Hierarchical depth")
    poincare_radius: float = Field(..., description="Hyperbolic radius")
    embedding_norm: float = Field(..., description="Embedding norm (scalar)")
    cluster_idx: int = Field(..., description="Cluster assignment index")
    confidence: float = Field(..., description="Cluster confidence")
    margin: float = Field(..., description="Decision margin")


class VisualizationConfig(BaseModel):
    """Visualization configuration."""

    depth_colors: List[str] = Field(..., min_length=10, max_length=10, description="Depth-to-color mapping")
    amino_acid_colors: Dict[str, str] = Field(..., description="Amino acid to color mapping")
    fiber_threshold: float = Field(..., description="Fiber threshold for edge computation")


class VisualizationResponse(BaseModel):
    """Full visualization payload."""

    points: List[CodonPoint] = Field(..., description="All 64 codon points")
    edges: List[Edge] = Field(..., description="Edge connections")
    config: VisualizationConfig = Field(..., description="Visualization configuration")
    metadata: ModelMetadata = Field(..., description="Model metadata")


class DepthStats(BaseModel):
    """Single depth level statistics."""

    intra_aa_variance: float = Field(..., description="Variance within amino acid groups")
    inter_aa_variance: float = Field(..., description="Variance between amino acid centroids")
    ratio: float = Field(..., description="Ratio: inter/intra")
    amino_acids: List[str] = Field(..., description="Amino acids present at this depth")


class AngularVarianceSummary(BaseModel):
    """Summary statistics for angular variance."""

    mean_intra: float
    mean_inter: float
    overall_ratio: float


class AngularVarianceResponse(BaseModel):
    """Response from GET /api/angular_variance."""

    by_depth: Dict[int, DepthStats] = Field(..., description="Statistics per depth level")
    summary: AngularVarianceSummary = Field(..., description="Summary statistics")


class VariantStats(BaseModel):
    """Trajectory statistics for a variant."""

    mean_depth: float
    depth_std: float
    mean_confidence: float


class SynonymousVariant(BaseModel):
    """Single synonymous variant."""

    strategy: Literal["random", "high_depth", "low_depth"] = Field(..., description="Strategy used")
    dna: str = Field(..., description="Generated DNA sequence")
    encoded: List[EncodedCodon] = Field(..., description="Encoded trajectory")
    stats: VariantStats = Field(..., description="Trajectory statistics")


class SynonymousVariantsResponse(BaseModel):
    """Response from POST /api/synonymous_variants."""

    protein: str = Field(..., description="Original protein sequence")
    variants: List[SynonymousVariant] = Field(..., description="Generated variants")


class SingleCodonResponse(BaseModel):
    """Response from GET /api/codon/{codon}."""

    codon: str
    amino_acid: str
    position: int
    depth: int
    projection: Tuple[float, float, float]
    poincare_radius: float
    cluster_idx: int
    confidence: float
    margin: float


class ClusterResponse(BaseModel):
    """Response from GET /api/cluster/{idx}."""

    cluster_idx: int
    amino_acid: str
    codons: List[SingleCodonResponse]


class DepthLevelResponse(BaseModel):
    """Response from GET /api/depth/{level}."""

    depth: int
    poincare_radius: float
    codons: List[SingleCodonResponse]


# =============================================================================
# Error Types
# =============================================================================


class APIError(BaseModel):
    """API error response."""

    detail: str
    status_code: int
