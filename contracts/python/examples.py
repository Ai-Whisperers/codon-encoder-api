"""
Codon Inference API - Python Usage Examples

These examples demonstrate how to consume the Codon Encoder inference service
using the Python client. No model weights or raw embeddings are exposed.

Requirements:
    pip install httpx pydantic

Usage:
    python examples.py
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

# Import from the local package (adjust path as needed)
try:
    from . import CodonClient, encode_sequence
except ImportError:
    # For direct execution
    from client import CodonClient
    from models import EncodedCodon


def example_basic_encoding():
    """
    Example 1: Basic sequence encoding

    Encode a DNA sequence and get inference results for each codon.
    """
    print("=" * 60)
    print("Example 1: Basic Sequence Encoding")
    print("=" * 60)

    client = CodonClient()

    # Encode a short DNA sequence
    sequence = "ATGGCTCTGTGG"
    result = client.encode(sequence)

    print(f"Sequence: {sequence}")
    print(f"Codons found: {len(result)}")
    print()

    for codon in result:
        print(
            f"  {codon.codon} -> {codon.amino_acid:3s} | "
            f"depth={codon.depth} | "
            f"conf={codon.confidence:.2f} | "
            f"margin={codon.margin:.3f}"
        )


def example_synonymous_variants():
    """
    Example 2: Generate synonymous DNA variants

    Create alternative DNA sequences encoding the same protein,
    using different codon selection strategies.
    """
    print("\n" + "=" * 60)
    print("Example 2: Synonymous Variants")
    print("=" * 60)

    client = CodonClient()

    # Generate variants for a short peptide
    protein = "MDDII"
    result = client.get_synonymous_variants(protein, n_variants=3)

    print(f"Protein: {result.protein}")
    print()

    for variant in result.variants:
        print(f"Strategy: {variant.strategy}")
        print(f"  DNA: {variant.dna}")
        print(f"  Mean depth: {variant.stats.mean_depth:.2f}")
        print(f"  Depth std: {variant.stats.depth_std:.2f}")
        print(f"  Mean confidence: {variant.stats.mean_confidence:.2f}")
        print()


def example_lookup_codon():
    """
    Example 3: Look up a specific codon

    Get detailed inference results for a single codon.
    """
    print("=" * 60)
    print("Example 3: Codon Lookup")
    print("=" * 60)

    client = CodonClient()

    # Look up the start codon
    codon_info = client.get_codon("ATG")

    print(f"Codon: {codon_info.codon}")
    print(f"  Amino acid: {codon_info.amino_acid}")
    print(f"  Position: {codon_info.position}")
    print(f"  Depth: {codon_info.depth}")
    print(f"  Poincaré radius: {codon_info.poincare_radius:.3f}")
    print(f"  Cluster: {codon_info.cluster_idx}")
    print(f"  Confidence: {codon_info.confidence:.3f}")
    print(f"  Margin: {codon_info.margin:.3f}")
    print(f"  Projection: ({codon_info.projection[0]:.3f}, "
          f"{codon_info.projection[1]:.3f}, {codon_info.projection[2]:.3f})")


def example_cluster_analysis():
    """
    Example 4: Analyze an amino acid cluster

    Get all synonymous codons for a specific amino acid.
    """
    print("\n" + "=" * 60)
    print("Example 4: Cluster Analysis")
    print("=" * 60)

    client = CodonClient()

    # Get cluster for Leucine (typically cluster 0 or varies by model)
    cluster = client.get_cluster(0)

    print(f"Cluster {cluster.cluster_idx}: {cluster.amino_acid}")
    print(f"  Synonymous codons: {len(cluster.codons)}")
    print()

    for c in cluster.codons:
        print(f"    {c.codon}: depth={c.depth}, conf={c.confidence:.2f}")


def example_depth_stratification():
    """
    Example 5: Analyze codons by depth level

    Get all codons at a specific hierarchical depth.
    """
    print("\n" + "=" * 60)
    print("Example 5: Depth Stratification")
    print("=" * 60)

    client = CodonClient()

    # Get codons at depth 0 (outermost in Poincaré disk)
    depth_info = client.get_depth(0)

    print(f"Depth {depth_info.depth}")
    print(f"  Poincaré radius: {depth_info.poincare_radius:.3f}")
    print(f"  Codons at this depth: {len(depth_info.codons)}")
    print()

    # Group by amino acid
    by_aa: dict[str, list[str]] = {}
    for c in depth_info.codons:
        by_aa.setdefault(c.amino_acid, []).append(c.codon)

    for aa, codons in sorted(by_aa.items()):
        print(f"    {aa}: {', '.join(codons)}")


def example_angular_variance():
    """
    Example 6: Angular variance analysis

    Analyze the separation quality between amino acid groups.
    """
    print("\n" + "=" * 60)
    print("Example 6: Angular Variance Analysis")
    print("=" * 60)

    client = CodonClient()

    variance = client.get_angular_variance()

    print("Summary:")
    print(f"  Mean intra-AA variance: {variance.summary.mean_intra:.4f}")
    print(f"  Mean inter-AA variance: {variance.summary.mean_inter:.4f}")
    print(f"  Overall ratio: {variance.summary.overall_ratio:.2f}")
    print()

    print("By depth:")
    for depth, stats in sorted(variance.by_depth.items()):
        print(f"  Depth {depth}: ratio={stats.ratio:.2f}, "
              f"AAs=[{', '.join(stats.amino_acids[:3])}...]")


def example_visualization_data():
    """
    Example 7: Get full visualization data

    Retrieve the complete dataset for rendering.
    """
    print("\n" + "=" * 60)
    print("Example 7: Visualization Data")
    print("=" * 60)

    client = CodonClient()

    viz = client.get_visualization()

    print(f"Points: {len(viz.points)}")
    print(f"Edges: {len(viz.edges)}")
    print(f"Model version: {viz.metadata.version}")
    print(f"Structure score: {viz.metadata.structure_score:.4f}")
    print(f"Cluster accuracy: {viz.metadata.cluster_accuracy:.2%}")
    print(f"Projection method: {viz.metadata.projection_method}")


def example_quick_encode():
    """
    Example 8: Quick encoding function

    Use the convenience function for one-off encoding.
    """
    print("\n" + "=" * 60)
    print("Example 8: Quick Encode Function")
    print("=" * 60)

    # One-liner encoding
    codons = encode_sequence("ATGGAAGAGTGA")

    print(f"Encoded {len(codons)} codons:")
    for c in codons:
        aa = c.amino_acid if c.amino_acid != "*" else "STOP"
        print(f"  {c.codon} = {aa}")


def example_local_development():
    """
    Example 9: Connect to local server

    Use a custom base URL for local development.
    """
    print("\n" + "=" * 60)
    print("Example 9: Local Development")
    print("=" * 60)

    # Connect to local server
    client = CodonClient(base_url="http://localhost:8000")

    # This would work if running the visualizer server locally
    print("Connecting to local server at http://localhost:8000")
    print("(Skipping actual request - run visualizer locally to test)")


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    print()
    print("Codon Inference API - Python Client Examples")
    print("Note: These examples require the API server to be running.")
    print("      For local testing, start the visualizer server first.")
    print()

    # Uncomment to run examples against a running server:
    # example_basic_encoding()
    # example_synonymous_variants()
    # example_lookup_codon()
    # example_cluster_analysis()
    # example_depth_stratification()
    # example_angular_variance()
    # example_visualization_data()
    # example_quick_encode()
    # example_local_development()

    print("Examples are defined but not executed.")
    print("Uncomment the function calls above to run against a live server.")
