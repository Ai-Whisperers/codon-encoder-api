"""
Codon Inference API - Python Client

Open-source client for consuming the Codon Encoder inference service.
Model weights and raw embeddings are not exposed through this API.

API endpoint: https://codon.ai-whisperers.org/api

Example:
    >>> from codon_client import CodonClient, encode_sequence
    >>>
    >>> # Quick encoding
    >>> codons = encode_sequence("ATGGCTCTGTGG")
    >>>
    >>> # Full client
    >>> client = CodonClient()
    >>> result = client.encode("ATGGCTCTGTGG")
    >>> metadata = client.get_metadata()
"""

from .client import CodonClient, CodonClientError, encode_sequence
from .models import (
    AngularVarianceResponse,
    AngularVarianceSummary,
    ClusterResponse,
    CodonPoint,
    DepthLevelResponse,
    DepthStats,
    Edge,
    EncodeRequest,
    EncodedCodon,
    ModelMetadata,
    SingleCodonResponse,
    SynonymousVariant,
    SynonymousVariantsRequest,
    SynonymousVariantsResponse,
    VariantStats,
    VisualizationConfig,
    VisualizationResponse,
)

__all__ = [
    # Client
    "CodonClient",
    "CodonClientError",
    "encode_sequence",
    # Core models
    "CodonPoint",
    "Edge",
    "ModelMetadata",
    # Request models
    "EncodeRequest",
    "SynonymousVariantsRequest",
    # Response models
    "EncodedCodon",
    "VisualizationResponse",
    "VisualizationConfig",
    "AngularVarianceResponse",
    "AngularVarianceSummary",
    "DepthStats",
    "SynonymousVariant",
    "SynonymousVariantsResponse",
    "VariantStats",
    "SingleCodonResponse",
    "ClusterResponse",
    "DepthLevelResponse",
]

__version__ = "0.1.0"
