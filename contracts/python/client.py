"""
Codon Inference API - Python Client

API endpoint: https://codon.ai-whisperers.org/api

This client provides a type-safe interface for consuming the Codon Encoder
inference service. Model weights and raw embeddings are not exposed.

Example:
    >>> from codon_client import CodonClient
    >>> client = CodonClient()
    >>> result = client.encode("ATGGCTCTGTGG")
    >>> for codon in result:
    ...     print(f"{codon.codon} -> {codon.amino_acid} (depth={codon.depth})")
"""

from __future__ import annotations

from typing import List, Literal, Optional

import httpx

from .models import (
    AngularVarianceResponse,
    ClusterResponse,
    CodonPoint,
    DepthLevelResponse,
    Edge,
    EncodedCodon,
    ModelMetadata,
    SingleCodonResponse,
    SynonymousVariantsResponse,
    VisualizationResponse,
)


class CodonClientError(Exception):
    """Base exception for Codon API client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CodonClient:
    """
    Client for the Codon Inference API.

    Provides type-safe access to all API endpoints with automatic
    response validation.

    Args:
        base_url: API base URL (default: https://codon.ai-whisperers.org)
        timeout: Request timeout in seconds (default: 30)
        api_key: Optional API key for authentication

    Example:
        >>> client = CodonClient()
        >>> metadata = client.get_metadata()
        >>> print(f"Model version: {metadata.version}")
    """

    DEFAULT_BASE_URL = "https://codon.ai-whisperers.org"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        api_key: str | None = None,
    ):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._headers = {}
        if api_key:
            self._headers["X-API-Key"] = api_key

    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}/api{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    headers=self._headers,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise CodonClientError(
                f"API error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise CodonClientError(f"Request failed: {e}") from e

    # =========================================================================
    # Core Endpoints
    # =========================================================================

    def get_visualization(self) -> VisualizationResponse:
        """
        Get complete visualization data.

        Returns all 64 codon points, edges, configuration, and metadata.

        Returns:
            VisualizationResponse with points, edges, config, and metadata
        """
        data = self._request("GET", "/visualization")
        return VisualizationResponse.model_validate(data)

    def get_points(self) -> List[CodonPoint]:
        """
        Get all 64 codon points with inference results.

        Returns:
            List of CodonPoint objects
        """
        data = self._request("GET", "/points")
        return [CodonPoint.model_validate(p) for p in data]

    def get_edges(
        self,
        mode: Literal["hierarchical", "amino_acid", "depth", "none"] = "hierarchical",
    ) -> List[Edge]:
        """
        Get fiber/edge connections between codons.

        Args:
            mode: Connection mode
                - hierarchical: same depth, embedding distance < threshold
                - amino_acid: synonymous codons (same translation)
                - depth: adjacent depth levels
                - none: no edges

        Returns:
            List of Edge objects
        """
        data = self._request("GET", "/edges", params={"mode": mode})
        return [Edge.model_validate(e) for e in data]

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            ModelMetadata with version, scores, and configuration
        """
        data = self._request("GET", "/metadata")
        return ModelMetadata.model_validate(data)

    # =========================================================================
    # Inference Endpoints
    # =========================================================================

    def encode(self, sequence: str) -> List[EncodedCodon]:
        """
        Encode a DNA sequence into codon trajectory.

        Takes a DNA sequence and returns inference results for each codon,
        including 3D projections, cluster assignments, and confidence scores.

        Args:
            sequence: DNA sequence (ATCG characters, non-ATCG filtered)

        Returns:
            List of EncodedCodon objects, one per codon in the sequence

        Example:
            >>> result = client.encode("ATGGCTCTGTGG")
            >>> for codon in result:
            ...     print(f"{codon.codon}: depth={codon.depth}, conf={codon.confidence:.2f}")
        """
        data = self._request("POST", "/encode", json={"sequence": sequence})
        return [EncodedCodon.model_validate(c) for c in data]

    def get_synonymous_variants(
        self,
        protein: str,
        n_variants: int = 3,
    ) -> SynonymousVariantsResponse:
        """
        Generate synonymous DNA variants for a protein sequence.

        Creates alternative DNA sequences that encode the same protein
        using different codon choices, with various selection strategies.

        Args:
            protein: Protein sequence (single-letter amino acid codes)
            n_variants: Number of variants to generate (default: 3)

        Returns:
            SynonymousVariantsResponse with variants using different strategies:
                - random: random codon selection
                - high_depth: prefer inner codons (deeper in hierarchy)
                - low_depth: prefer outer codons (shallower in hierarchy)

        Example:
            >>> result = client.get_synonymous_variants("MDDII", n_variants=3)
            >>> for v in result.variants:
            ...     print(f"{v.strategy}: {v.dna} (mean_depth={v.stats.mean_depth:.2f})")
        """
        data = self._request(
            "POST",
            "/synonymous_variants",
            json={"protein": protein, "n_variants": n_variants},
        )
        return SynonymousVariantsResponse.model_validate(data)

    # =========================================================================
    # Lookup Endpoints
    # =========================================================================

    def get_codon(self, codon: str) -> SingleCodonResponse:
        """
        Get inference results for a single codon.

        Args:
            codon: Three-letter codon (e.g., "ATG")

        Returns:
            SingleCodonResponse with all inference metrics
        """
        data = self._request("GET", f"/codon/{codon.upper()}")
        return SingleCodonResponse.model_validate(data)

    def get_cluster(self, idx: int) -> ClusterResponse:
        """
        Get all codons in an amino acid cluster.

        Args:
            idx: Cluster index (0-20)

        Returns:
            ClusterResponse with cluster info and member codons
        """
        data = self._request("GET", f"/cluster/{idx}")
        return ClusterResponse.model_validate(data)

    def get_depth(self, level: int) -> DepthLevelResponse:
        """
        Get all codons at a specific depth level.

        Args:
            level: Depth level (0-9)

        Returns:
            DepthLevelResponse with codons at that depth
        """
        data = self._request("GET", f"/depth/{level}")
        return DepthLevelResponse.model_validate(data)

    # =========================================================================
    # Analytics Endpoints
    # =========================================================================

    def get_angular_variance(self) -> AngularVarianceResponse:
        """
        Get angular variance statistics by depth level.

        Returns circular variance metrics measuring the separation
        quality between and within amino acid groups at each depth.

        Returns:
            AngularVarianceResponse with per-depth and summary statistics
        """
        data = self._request("GET", "/angular_variance")
        return AngularVarianceResponse.model_validate(data)

    def get_reference_actb(self) -> SynonymousVariantsResponse:
        """
        Get reference ACTB gene with synonymous variants.

        Returns the first 60 amino acids of beta-actin (ACTB)
        with three encoded variants for comparison.

        Returns:
            SynonymousVariantsResponse with ACTB variants
        """
        data = self._request("GET", "/reference/actb")
        return SynonymousVariantsResponse.model_validate(data)


# Convenience function for quick encoding
def encode_sequence(
    sequence: str,
    base_url: str | None = None,
) -> List[EncodedCodon]:
    """
    Quick function to encode a DNA sequence.

    Args:
        sequence: DNA sequence
        base_url: Optional API base URL

    Returns:
        List of EncodedCodon objects
    """
    client = CodonClient(base_url=base_url)
    return client.encode(sequence)
