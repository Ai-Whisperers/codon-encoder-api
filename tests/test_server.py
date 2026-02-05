"""
Tests for visualizer/server.py API endpoints.

Tests all FastAPI endpoints including:
- GET /api/visualization
- POST /api/encode
- POST /api/encode/batch
- GET /api/codon/{codon}
- GET /api/cluster/{idx}
- GET /api/depth/{level}
- GET /api/angular_variance
- POST /api/synonymous_variants
- GET /api/reference/actb
"""



class TestVisualizationEndpoint:
    """Tests for GET /api/visualization."""

    def test_get_visualization(self, test_client):
        """Should return complete visualization data."""
        response = test_client.get("/api/visualization")
        assert response.status_code == 200

        data = response.json()
        assert "points" in data
        assert "edges" in data
        assert "metadata" in data
        assert "config" in data

    def test_visualization_has_64_points(self, test_client):
        """Should return exactly 64 codon points."""
        response = test_client.get("/api/visualization")
        data = response.json()
        assert len(data["points"]) == 64

    def test_visualization_point_structure(self, test_client):
        """Each point should have required fields."""
        response = test_client.get("/api/visualization")
        data = response.json()

        for point in data["points"]:
            assert "codon" in point
            assert "amino_acid" in point
            assert "depth" in point
            assert "projection" in point
            assert "confidence" in point
            assert "margin" in point
            assert len(point["projection"]) == 3


class TestPointsEndpoint:
    """Tests for GET /api/points."""

    def test_get_points(self, test_client):
        """Should return codon points only."""
        response = test_client.get("/api/points")
        assert response.status_code == 200

        points = response.json()
        assert len(points) == 64

    def test_points_are_list(self, test_client):
        """Response should be a list."""
        response = test_client.get("/api/points")
        assert isinstance(response.json(), list)


class TestEdgesEndpoint:
    """Tests for GET /api/edges."""

    def test_get_edges_hierarchical(self, test_client):
        """Hierarchical mode should return edges."""
        response = test_client.get("/api/edges?mode=hierarchical")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_edges_amino_acid(self, test_client):
        """Amino acid mode should return synonymous connections."""
        response = test_client.get("/api/edges?mode=amino_acid")
        assert response.status_code == 200
        edges = response.json()
        assert len(edges) > 0  # Should have many synonymous edges

    def test_get_edges_none(self, test_client):
        """None mode should return empty list."""
        response = test_client.get("/api/edges?mode=none")
        assert response.status_code == 200
        assert response.json() == []


class TestMetadataEndpoint:
    """Tests for GET /api/metadata."""

    def test_get_metadata(self, test_client):
        """Should return model metadata."""
        response = test_client.get("/api/metadata")
        assert response.status_code == 200

        metadata = response.json()
        assert "n_codons" in metadata
        assert metadata["n_codons"] == 64


class TestEncodeEndpoint:
    """Tests for POST /api/encode."""

    def test_encode_valid_sequence(self, test_client):
        """Should encode valid DNA sequence."""
        response = test_client.post("/api/encode", json={"sequence": "ATGGCTCTGTGG"})
        assert response.status_code == 200

        result = response.json()
        assert len(result) == 4  # 4 codons

    def test_encode_returns_correct_codons(self, test_client):
        """Should return correct codon info."""
        response = test_client.post("/api/encode", json={"sequence": "ATGGCT"})
        result = response.json()

        assert result[0]["codon"] == "ATG"
        assert result[0]["amino_acid"] == "M"
        assert result[1]["codon"] == "GCT"
        assert result[1]["amino_acid"] == "A"

    def test_encode_empty_sequence_400(self, test_client):
        """Empty sequence should return 400."""
        response = test_client.post("/api/encode", json={"sequence": ""})
        assert response.status_code == 400

    def test_encode_too_short_400(self, test_client):
        """Sequence < 3 nucleotides should return 400."""
        response = test_client.post("/api/encode", json={"sequence": "AT"})
        assert response.status_code == 400

    def test_encode_strips_invalid_chars(self, test_client):
        """Invalid characters should be stripped."""
        response = test_client.post("/api/encode", json={"sequence": "ATG---GCT"})
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2

    def test_encode_handles_lowercase(self, test_client):
        """Lowercase should work."""
        response = test_client.post("/api/encode", json={"sequence": "atggct"})
        assert response.status_code == 200
        result = response.json()
        assert result[0]["amino_acid"] == "M"

    def test_encode_codon_structure(self, test_client):
        """Each encoded codon should have required fields."""
        response = test_client.post("/api/encode", json={"sequence": "ATG"})
        result = response.json()

        codon = result[0]
        assert "seq_position" in codon
        assert "codon" in codon
        assert "amino_acid" in codon
        assert "depth" in codon
        assert "confidence" in codon
        assert "margin" in codon


class TestBatchEncodeEndpoint:
    """Tests for POST /api/encode/batch."""

    def test_encode_batch_json(self, test_client, batch_input_json):
        """Should batch encode JSON format."""
        response = test_client.post("/api/encode/batch", json=batch_input_json)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "summary" in data
        assert data["summary"]["successful"] == 3

    def test_encode_batch_fasta(self, test_client, batch_input_fasta):
        """Should batch encode FASTA format."""
        response = test_client.post("/api/encode/batch", json=batch_input_fasta)
        assert response.status_code == 200

        data = response.json()
        assert data["summary"]["successful"] == 3

    def test_encode_batch_csv(self, test_client, batch_input_csv):
        """Should batch encode CSV format."""
        response = test_client.post("/api/encode/batch", json=batch_input_csv)
        assert response.status_code == 200

        data = response.json()
        assert data["summary"]["successful"] == 3

    def test_encode_batch_tsv(self, test_client):
        """Should batch encode TSV format."""
        tsv_input = {
            "format": "tsv",
            "sequences": "id\tsequence\nseq1\tATGGCT\nseq2\tATGAAA",
        }
        response = test_client.post("/api/encode/batch", json=tsv_input)
        assert response.status_code == 200

        data = response.json()
        assert data["summary"]["successful"] == 2

    def test_encode_batch_includes_protein(self, test_client, batch_input_json):
        """Batch results should include protein translation."""
        response = test_client.post("/api/encode/batch", json=batch_input_json)
        data = response.json()

        for result in data["results"]:
            assert "protein" in result

    def test_encode_batch_includes_stats(self, test_client, batch_input_json):
        """Batch results should include statistics."""
        response = test_client.post("/api/encode/batch", json=batch_input_json)
        data = response.json()

        for result in data["results"]:
            stats = result["stats"]
            assert "length" in stats
            assert "mean_depth" in stats
            assert "mean_confidence" in stats

    def test_encode_batch_error_handling(self, test_client):
        """Should handle invalid sequences gracefully."""
        batch = {
            "format": "json",
            "sequences": [
                {"id": "valid", "sequence": "ATGGCT"},
                {"id": "too_short", "sequence": "AT"},  # Invalid
            ],
        }
        response = test_client.post("/api/encode/batch", json=batch)
        assert response.status_code == 200

        data = response.json()
        assert data["summary"]["successful"] == 1
        assert data["summary"]["failed_sequences"] == 1
        assert len(data["errors"]) == 1

    def test_encode_batch_unknown_format_400(self, test_client):
        """Unknown format should return 400."""
        response = test_client.post(
            "/api/encode/batch",
            json={"format": "xml", "sequences": "test"},
        )
        assert response.status_code == 400


class TestCodonLookupEndpoint:
    """Tests for GET /api/codon/{codon}."""

    def test_codon_lookup_valid(self, test_client):
        """Should return codon info for valid codon."""
        response = test_client.get("/api/codon/ATG")
        assert response.status_code == 200

        codon = response.json()
        assert codon["codon"] == "ATG"
        assert codon["amino_acid"] == "M"

    def test_codon_lookup_lowercase(self, test_client):
        """Lowercase codon should work."""
        response = test_client.get("/api/codon/atg")
        assert response.status_code == 200

    def test_codon_lookup_stop_codon(self, test_client):
        """Stop codons should work."""
        response = test_client.get("/api/codon/TAA")
        assert response.status_code == 200

        codon = response.json()
        assert codon["amino_acid"] == "*"

    def test_codon_lookup_invalid_404(self, test_client):
        """Invalid codon should return 404."""
        response = test_client.get("/api/codon/XYZ")
        assert response.status_code == 404


class TestClusterEndpoint:
    """Tests for GET /api/cluster/{idx}."""

    def test_cluster_lookup_valid(self, test_client):
        """Should return cluster info for valid index."""
        response = test_client.get("/api/cluster/0")
        assert response.status_code == 200

        cluster = response.json()
        assert "cluster_idx" in cluster
        assert "amino_acid" in cluster
        assert "codons" in cluster

    def test_cluster_codons_same_cluster_idx(self, test_client):
        """All codons in cluster should have same cluster_idx."""
        response = test_client.get("/api/cluster/0")
        cluster = response.json()

        # All codons returned should have cluster_idx matching the request
        for codon in cluster["codons"]:
            assert codon["cluster_idx"] == 0

    def test_cluster_invalid_404(self, test_client):
        """Invalid cluster index should return 404."""
        response = test_client.get("/api/cluster/99")
        assert response.status_code == 404


class TestDepthEndpoint:
    """Tests for GET /api/depth/{level}."""

    def test_depth_lookup_valid(self, test_client):
        """Should return codons at depth level."""
        response = test_client.get("/api/depth/0")
        assert response.status_code == 200

        data = response.json()
        assert data["depth"] == 0
        assert "count" in data
        assert "codons" in data

    def test_depth_codons_have_correct_depth(self, test_client):
        """All codons should have the requested depth."""
        for level in range(10):
            response = test_client.get(f"/api/depth/{level}")
            data = response.json()

            for codon in data["codons"]:
                assert codon["depth"] == level

    def test_depth_out_of_range_400(self, test_client):
        """Depth > 9 should return 400."""
        response = test_client.get("/api/depth/10")
        assert response.status_code == 400

    def test_depth_negative_400(self, test_client):
        """Negative depth should return 400."""
        response = test_client.get("/api/depth/-1")
        assert response.status_code == 400


class TestAngularVarianceEndpoint:
    """Tests for GET /api/angular_variance."""

    def test_angular_variance(self, test_client):
        """Should return angular variance stats."""
        response = test_client.get("/api/angular_variance")
        assert response.status_code == 200

        data = response.json()
        assert "by_depth" in data
        assert "summary" in data

    def test_angular_variance_has_depth_entries(self, test_client):
        """Should have stats for available depth levels."""
        response = test_client.get("/api/angular_variance")
        data = response.json()

        # Check at least some depth levels are present
        assert len(data["by_depth"]) >= 1

        # Check each entry has required structure
        for depth_key, depth_data in data["by_depth"].items():
            assert "depth" in depth_data
            assert "n_codons" in depth_data
            assert "intra_aa_variance" in depth_data
            assert "inter_aa_variance" in depth_data


class TestSynonymousVariantsEndpoint:
    """Tests for POST /api/synonymous_variants."""

    def test_synonymous_variants(self, test_client):
        """Should generate synonymous variants."""
        response = test_client.post(
            "/api/synonymous_variants",
            json={"protein": "MDDII", "n_variants": 3},
        )
        assert response.status_code == 200

        variants = response.json()
        assert len(variants) == 3

    def test_synonymous_variants_capped(self, test_client):
        """Variants should be capped at 5."""
        response = test_client.post(
            "/api/synonymous_variants",
            json={"protein": "MA", "n_variants": 10},
        )
        assert response.status_code == 200

        variants = response.json()
        assert len(variants) == 5  # Capped at 5

    def test_synonymous_variants_empty_400(self, test_client):
        """Empty protein should return 400."""
        response = test_client.post(
            "/api/synonymous_variants",
            json={"protein": "", "n_variants": 3},
        )
        assert response.status_code == 400


class TestActbReferenceEndpoint:
    """Tests for GET /api/reference/actb."""

    def test_actb_reference(self, test_client):
        """Should return ACTB reference with variants."""
        response = test_client.get("/api/reference/actb")
        assert response.status_code == 200

        data = response.json()
        assert data["gene"] == "ACTB"
        assert "protein" in data
        assert "variants" in data
        assert len(data["variants"]) == 3


class TestRootEndpoint:
    """Tests for GET / (static file serving)."""

    def test_root_serves_html(self, test_client):
        """Root should serve index.html."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
