"""
Pytest configuration and shared fixtures for Codon Encoder API tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.constants import ALL_64_CODONS, CODON_TABLE


# =============================================================================
# MODEL FIXTURES
# =============================================================================


class DummyCodonEncoder(nn.Module):
    """Minimal CodonEncoder for testing (matches production architecture)."""

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
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.cluster_head(embedding)
        return embedding, logits

    def encode(self, x):
        return self.encoder(x)


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory) -> Path:
    """
    Create a dummy model checkpoint for testing.

    This fixture creates a temporary model file that persists for the entire
    test session, avoiding repeated model generation.
    """
    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "test_codon_encoder.pt"

    # Create model and mappings
    model = DummyCodonEncoder()

    codon_to_position = {c: i for i, c in enumerate(ALL_64_CODONS)}
    unique_aas = sorted(set(CODON_TABLE.values()))
    aa_to_cluster = {aa: i for i, aa in enumerate(unique_aas)}

    checkpoint = {
        "model_state": model.state_dict(),
        "codon_to_position": codon_to_position,
        "aa_to_cluster": aa_to_cluster,
        "metadata": {
            "version": "test-0.0.1",
            "hierarchy_correlation": 0.5,
            "cluster_accuracy": 0.95,
            "synonymous_accuracy": 0.90,
        },
    }

    torch.save(checkpoint, model_path)
    return model_path


@pytest.fixture
def model_loader(dummy_model_path):
    """
    Pre-loaded ModelLoader instance for testing.

    Returns a ModelLoader that has already loaded the dummy model.
    """
    from visualizer.model_loader import ModelLoader

    loader = ModelLoader(dummy_model_path)
    loader.load()
    return loader


# =============================================================================
# API CLIENT FIXTURES
# =============================================================================


@pytest.fixture
def test_client(dummy_model_path, monkeypatch):
    """
    FastAPI TestClient with dummy model loaded.

    Uses monkeypatch to set the model path before importing the app.
    """
    # Set environment variable before importing server
    monkeypatch.setenv("CODON_MODEL_PATH", str(dummy_model_path))

    from fastapi.testclient import TestClient
    from visualizer.server import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def async_client(dummy_model_path, monkeypatch):
    """
    Async HTTP client for testing async endpoints.
    """
    import httpx

    monkeypatch.setenv("CODON_MODEL_PATH", str(dummy_model_path))

    from visualizer.server import app

    return httpx.AsyncClient(app=app, base_url="http://test")


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================


@pytest.fixture
def valid_dna_sequences() -> dict[str, str]:
    """Collection of valid DNA sequences for testing."""
    return {
        "start_codon": "ATG",
        "short": "ATGGCTCTGTGG",  # M-A-L-W
        "with_stop": "ATGTAATAG",  # M-*-* (two stop codons)
        "long": "ATG" + "GCT" * 100 + "TAA",  # M + 100 Alanines + Stop
        "all_bases": "ATCGATCGATCG",  # Uses all 4 bases
        "lowercase": "atggctctgtgg",  # Should be normalized
        "with_spaces": "ATG GCT CTG TGG",  # Should be cleaned
    }


@pytest.fixture
def invalid_dna_sequences() -> dict[str, str]:
    """Collection of invalid DNA sequences for testing."""
    return {
        "empty": "",
        "too_short": "AT",  # Less than 3 nucleotides
        "invalid_chars": "ATGXYZCTG",  # Contains invalid characters
        "numbers": "ATG123CTG",  # Contains numbers
        "only_invalid": "XXXYYY",  # No valid nucleotides
    }


@pytest.fixture
def valid_proteins() -> dict[str, str]:
    """Collection of valid protein sequences for testing."""
    return {
        "simple": "MQ",
        "short": "MDDII",
        "actb_partial": "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQS",
        "with_stop": "MDDII*",
        "all_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
    }


@pytest.fixture
def batch_input_json() -> dict:
    """Sample batch input in JSON format."""
    return {
        "format": "json",
        "sequences": [
            {"id": "seq1", "sequence": "ATGGCTCTGTGG", "description": "Test sequence 1"},
            {"id": "seq2", "sequence": "ATGAAAGGG", "description": "Test sequence 2"},
            {"id": "seq3", "sequence": "ATGTAA"},  # With stop codon
        ],
    }


@pytest.fixture
def batch_input_fasta() -> dict:
    """Sample batch input in FASTA format."""
    fasta_content = """>seq1 Test sequence 1
ATGGCTCTGTGG
>seq2 Test sequence 2
ATGAAAGGG
>seq3
ATGTAA
"""
    return {"format": "fasta", "sequences": fasta_content}


@pytest.fixture
def batch_input_csv() -> dict:
    """Sample batch input in CSV format."""
    csv_content = """id,sequence,description
seq1,ATGGCTCTGTGG,Test sequence 1
seq2,ATGAAAGGG,Test sequence 2
seq3,ATGTAA,
"""
    return {"format": "csv", "sequences": csv_content}


# =============================================================================
# CODON DATA FIXTURES
# =============================================================================


@pytest.fixture
def all_codons() -> list[str]:
    """List of all 64 standard codons."""
    return ALL_64_CODONS.copy()


@pytest.fixture
def codon_table() -> dict[str, str]:
    """Standard genetic code mapping."""
    return CODON_TABLE.copy()


@pytest.fixture
def stop_codons() -> list[str]:
    """List of stop codons."""
    return ["TAA", "TAG", "TGA"]


@pytest.fixture
def start_codon() -> str:
    """The standard start codon."""
    return "ATG"


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    import logging

    # Clear all handlers from codon loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("codon"):
            logger = logging.getLogger(name)
            logger.handlers.clear()
    yield


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
