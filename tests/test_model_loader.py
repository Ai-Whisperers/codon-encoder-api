"""
Tests for visualizer/model_loader.py

Tests the core inference engine including:
- Codon encoding (one-hot)
- Depth level computation
- Poincare radius mapping
- Model loading and embedding extraction
- Sequence encoding
- Synonymous variant generation
"""

import numpy as np
import pytest

from server.constants import ALL_64_CODONS, CODON_TABLE, BASE_TO_IDX
from visualizer.model_loader import (
    ModelLoader,
    codon_to_onehot,
    compute_depth_level,
    depth_to_poincare_radius,
    validate_translation,
)


class TestCodonToOnehot:
    """Tests for codon_to_onehot function."""

    def test_start_codon_atg(self):
        """ATG should produce correct one-hot encoding."""
        onehot = codon_to_onehot("ATG")
        assert onehot.shape == (12,)
        assert onehot.sum() == 3.0  # Exactly 3 ones

        # ATG: A at pos 0, T at pos 1, G at pos 2
        # Position 0 (A): index 2 in ATCG
        # Position 1 (T): index 0 in ATCG
        # Position 2 (G): index 3 in ATCG
        expected_indices = [
            0 * 4 + BASE_TO_IDX["A"],  # Position 0, base A
            1 * 4 + BASE_TO_IDX["T"],  # Position 1, base T
            2 * 4 + BASE_TO_IDX["G"],  # Position 2, base G
        ]
        for idx in expected_indices:
            assert onehot[idx] == 1.0, f"Expected 1.0 at index {idx}"

    def test_all_64_codons_valid(self):
        """All standard codons should produce valid one-hot vectors."""
        for codon in ALL_64_CODONS:
            onehot = codon_to_onehot(codon)
            assert onehot.shape == (12,), f"Wrong shape for {codon}"
            assert onehot.sum() == 3.0, f"Wrong sum for {codon}: {onehot.sum()}"
            assert np.all((onehot == 0) | (onehot == 1)), f"Non-binary values for {codon}"

    def test_lowercase_codons(self):
        """Lowercase codons should work the same as uppercase."""
        upper = codon_to_onehot("ATG")
        lower = codon_to_onehot("atg")
        np.testing.assert_array_equal(upper, lower)

    def test_mixed_case_codons(self):
        """Mixed case codons should work."""
        result = codon_to_onehot("AtG")
        assert result.sum() == 3.0

    def test_stop_codons(self):
        """Stop codons should produce valid encodings."""
        for stop in ["TAA", "TAG", "TGA"]:
            onehot = codon_to_onehot(stop)
            assert onehot.shape == (12,)
            assert onehot.sum() == 3.0

    def test_invalid_codon_returns_zeros(self):
        """Invalid codons should return all zeros (graceful degradation)."""
        onehot = codon_to_onehot("XYZ")
        assert onehot.shape == (12,)
        assert onehot.sum() == 0.0


class TestComputeDepthLevel:
    """Tests for compute_depth_level function."""

    def test_position_zero_is_max_depth(self):
        """Position 0 should return max depth (9)."""
        assert compute_depth_level(0) == 9

    def test_position_one_is_depth_zero(self):
        """Position 1 should return depth 0."""
        assert compute_depth_level(1) == 0

    def test_position_two_is_depth_zero(self):
        """Position 2 should return depth 0."""
        assert compute_depth_level(2) == 0

    def test_position_three_is_depth_one(self):
        """Position 3 (3^1) should return depth 1."""
        assert compute_depth_level(3) == 1

    def test_position_nine_is_depth_two(self):
        """Position 9 (3^2) should return depth 2."""
        assert compute_depth_level(9) == 2

    def test_position_27_is_depth_three(self):
        """Position 27 (3^3) should return depth 3."""
        assert compute_depth_level(27) == 3

    def test_depth_capped_at_max(self):
        """Depth should be capped at max_depth."""
        # Very large position should still be at most max_depth
        result = compute_depth_level(10000, max_depth=9)
        assert result <= 9

    def test_custom_max_depth(self):
        """Custom max_depth should work."""
        assert compute_depth_level(0, max_depth=5) == 5

    def test_all_positions_valid_depth(self):
        """All positions 0-63 should return valid depths."""
        for pos in range(64):
            depth = compute_depth_level(pos)
            assert 0 <= depth <= 9, f"Invalid depth {depth} for position {pos}"


class TestDepthToPoincare:
    """Tests for depth_to_poincare_radius function."""

    def test_depth_zero_is_outer(self):
        """Depth 0 (outer) should return 0.9."""
        assert abs(depth_to_poincare_radius(0) - 0.9) < 0.001

    def test_depth_nine_is_center(self):
        """Depth 9 (center) should return 0.1."""
        assert abs(depth_to_poincare_radius(9) - 0.1) < 0.001

    def test_depth_monotonic_decreasing(self):
        """Radius should decrease as depth increases."""
        prev_radius = depth_to_poincare_radius(0)
        for d in range(1, 10):
            current_radius = depth_to_poincare_radius(d)
            assert current_radius < prev_radius, f"Non-monotonic at depth {d}"
            prev_radius = current_radius

    def test_middle_depths(self):
        """Middle depths should have intermediate radii."""
        for d in range(10):
            r = depth_to_poincare_radius(d)
            # Use approximate comparison for float precision
            assert 0.1 - 1e-9 <= r <= 0.9 + 1e-9, f"Radius {r} out of range for depth {d}"


class TestValidateTranslation:
    """Tests for validate_translation function."""

    def test_simple_translation(self):
        """Simple translation should work."""
        assert validate_translation("ATGGCT", "MA")  # ATG=M, GCT=A

    def test_start_codon(self):
        """Start codon ATG should translate to M."""
        assert validate_translation("ATG", "M")

    def test_stop_codons(self):
        """Stop codons should translate to *."""
        assert validate_translation("TAA", "*")
        assert validate_translation("TAG", "*")
        assert validate_translation("TGA", "*")

    def test_full_sequence(self):
        """Longer sequence translation."""
        dna = "ATGGCTCTGTGG"  # M-A-L-W
        assert validate_translation(dna, "MALW")

    def test_with_stop_codon(self):
        """Sequence ending with stop codon."""
        assert validate_translation("ATGTAA", "M*")

    def test_invalid_translation_returns_false(self):
        """Wrong protein should return False."""
        assert not validate_translation("ATG", "X")
        assert not validate_translation("ATGGCT", "MM")


class TestModelLoader:
    """Tests for ModelLoader class."""

    def test_load_model(self, dummy_model_path):
        """Model should load successfully."""
        loader = ModelLoader(dummy_model_path).load()
        assert loader.model is not None
        assert loader.checkpoint is not None
        assert loader.cluster_centers is not None

    def test_load_nonexistent_raises(self, temp_directory):
        """Loading nonexistent model should raise FileNotFoundError."""
        fake_path = temp_directory / "nonexistent.pt"
        loader = ModelLoader(fake_path)
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_extract_embeddings(self, model_loader):
        """Should extract embeddings for all 64 codons."""
        # Ensure embeddings are extracted
        embeddings = model_loader.extract_embeddings()
        assert len(embeddings) == 64

        for codon in ALL_64_CODONS:
            assert codon in embeddings
            emb = embeddings[codon]
            assert emb.shape == (16,), f"Wrong embedding shape for {codon}"
            assert not np.any(np.isnan(emb)), f"NaN in embedding for {codon}"
            assert not np.any(np.isinf(emb)), f"Inf in embedding for {codon}"

    def test_compute_projection_pca(self, model_loader):
        """PCA projection should produce (64, 3) array."""
        proj = model_loader.compute_projection(method="pca")
        assert proj.shape == (64, 3)

        # Check values are finite and within reasonable range
        assert not np.any(np.isnan(proj)), "NaN in projections"
        assert not np.any(np.isinf(proj)), "Inf in projections"
        # Projections are scaled by max, so all values should be <= 1
        # (with small tolerance for floating point)
        assert np.abs(proj).max() <= 1.01, "Projections not properly scaled"

    def test_build_codon_points(self, model_loader):
        """Should build 64 valid CodonPoint objects."""
        # Ensure points are built
        model_loader._ensure_initialized()
        points = model_loader.codon_points
        assert len(points) == 64

        for point in points:
            assert point.codon in CODON_TABLE
            assert point.amino_acid in set(CODON_TABLE.values())
            assert len(point.projection) == 3
            assert 0 <= point.depth <= 9
            assert 0 <= point.confidence <= 1
            # Use approximate bounds for floating point precision
            assert 0.1 - 1e-9 <= point.poincare_radius <= 0.9 + 1e-9

    def test_encode_sequence_simple(self, model_loader):
        """Simple sequence encoding."""
        result = model_loader.encode_sequence("ATGGCTCTGTGG")
        assert len(result) == 4  # 4 codons

        assert result[0]["codon"] == "ATG"
        assert result[0]["amino_acid"] == "M"
        assert result[1]["amino_acid"] == "A"
        assert result[2]["amino_acid"] == "L"
        assert result[3]["amino_acid"] == "W"

    def test_encode_sequence_with_stop(self, model_loader):
        """Sequence with stop codon should stop at stop."""
        result = model_loader.encode_sequence("ATGTAA")
        assert len(result) == 2  # M + *
        assert result[-1]["amino_acid"] == "*"

    def test_encode_sequence_lowercase(self, model_loader):
        """Lowercase sequence should work."""
        result = model_loader.encode_sequence("atggct")
        assert len(result) == 2
        assert result[0]["amino_acid"] == "M"

    def test_encode_sequence_with_invalid_chars(self, model_loader):
        """Invalid characters should be stripped."""
        result = model_loader.encode_sequence("ATG---GCT")
        assert len(result) == 2

    def test_encode_sequence_depth_values(self, model_loader):
        """All encoded codons should have valid depth."""
        result = model_loader.encode_sequence("ATGGCTCTGTGGATGAAAGGG")
        for codon_data in result:
            assert 0 <= codon_data["depth"] <= 9

    def test_synonymous_variants_count(self, model_loader):
        """Should generate correct number of variants."""
        variants = model_loader.generate_synonymous_variants("MQ", n_variants=3)
        assert len(variants) == 3

    def test_synonymous_variants_strategies(self, model_loader):
        """Variants should use different strategies."""
        variants = model_loader.generate_synonymous_variants("MA", n_variants=3)
        strategies = {v["strategy"] for v in variants}
        assert "random" in strategies
        assert "high_depth" in strategies
        assert "low_depth" in strategies

    def test_synonymous_variants_encode_same_protein(self, model_loader):
        """All variants should encode the same protein."""
        protein = "MDDII"
        variants = model_loader.generate_synonymous_variants(protein, n_variants=3)

        for variant in variants:
            # Check DNA translates to same protein
            encoded = variant["encoded"]
            translated = "".join(c["amino_acid"] for c in encoded)
            assert translated == protein, f"Variant doesn't match: {translated} != {protein}"

    def test_compute_edges_hierarchical(self, model_loader):
        """Hierarchical edges should connect same-depth codons."""
        edges = model_loader.compute_edges(mode="hierarchical")
        # Edges exist (may be empty if threshold is very low)
        assert isinstance(edges, list)
        for edge in edges:
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge

    def test_compute_edges_amino_acid(self, model_loader):
        """Amino acid edges should connect synonymous codons."""
        edges = model_loader.compute_edges(mode="amino_acid")
        assert len(edges) > 0  # Should have many synonymous connections

        # Check edges connect codons with same amino acid
        points = model_loader.codon_points
        for edge in edges:
            source_aa = points[edge["source"]].amino_acid
            target_aa = points[edge["target"]].amino_acid
            assert source_aa == target_aa, "Edge connects different amino acids"

    def test_compute_edges_none(self, model_loader):
        """Mode 'none' should return empty list."""
        edges = model_loader.compute_edges(mode="none")
        assert edges == []

    def test_angular_variance_structure(self, model_loader):
        """Angular variance should have correct structure."""
        result = model_loader.compute_angular_variance()

        assert "by_depth" in result
        assert "summary" in result

        # Check at least some depths are present (dummy model may not have all 10)
        assert len(result["by_depth"]) >= 1

        # Check depth entries have correct structure
        for d, depth_data in result["by_depth"].items():
            assert "depth" in depth_data
            assert "n_codons" in depth_data
            assert "intra_aa_variance" in depth_data
            assert "inter_aa_variance" in depth_data

        # Check summary fields
        summary = result["summary"]
        assert "mean_intra_variance" in summary
        assert "mean_inter_variance" in summary
        assert "mean_ratio" in summary

    def test_get_visualization_data(self, model_loader):
        """Should return complete visualization payload."""
        data = model_loader.get_visualization_data()

        assert len(data.points) == 64
        assert isinstance(data.edges, list)
        assert isinstance(data.cluster_centers_3d, list)
        assert "model_path" in data.metadata
        assert "n_codons" in data.metadata
        assert data.metadata["n_codons"] == 64


class TestModelLoaderEdgeCases:
    """Edge case tests for ModelLoader."""

    def test_encode_empty_sequence(self, model_loader):
        """Empty sequence should return empty list."""
        result = model_loader.encode_sequence("")
        assert result == []

    def test_encode_too_short_sequence(self, model_loader):
        """Sequence shorter than 3 should return empty list."""
        result = model_loader.encode_sequence("AT")
        assert result == []

    def test_encode_partial_codon_at_end(self, model_loader):
        """Partial codon at end should be ignored."""
        result = model_loader.encode_sequence("ATGAT")  # ATG + partial AT
        assert len(result) == 1
        assert result[0]["codon"] == "ATG"

    def test_synonymous_variants_single_aa(self, model_loader):
        """Single amino acid should work."""
        variants = model_loader.generate_synonymous_variants("M", n_variants=2)
        assert len(variants) == 2
        # M only has one codon (ATG), so all variants should be the same
        for v in variants:
            assert v["dna_sequence"] == "ATG"

    def test_synonymous_variants_with_stop(self, model_loader):
        """Stop codon in protein should work."""
        variants = model_loader.generate_synonymous_variants("M*", n_variants=2)
        assert len(variants) == 2
