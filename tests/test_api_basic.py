"""
Basic API tests that don't require the model to be loaded.
These tests verify the API structure and basic functionality.
"""

from fastapi import FastAPI


def test_fastapi_app_creation():
    """Test that we can create a FastAPI app."""
    app = FastAPI(title="Test App")
    assert app.title == "Test App"


def test_dna_sequence_validation():
    """Test DNA sequence validation functions."""
    from visualizer.server import clean_sequence, translate_codon
    
    # Test sequence cleaning
    assert clean_sequence("ATCG") == "ATCG"
    assert clean_sequence("atcg") == "ATCG"
    assert clean_sequence("ATCG123XYZ") == "ATCG"
    assert clean_sequence("A-T-C-G") == "ATCG"
    assert clean_sequence("") == ""
    
    # Test codon translation
    assert translate_codon("ATG") == "M"
    assert translate_codon("atg") == "M"
    assert translate_codon("TAA") == "*"
    assert translate_codon("XXX") == "?"


def test_batch_format_parsing():
    """Test batch format parsing functions."""
    from visualizer.server import parse_fasta, parse_csv_tsv
    
    # Test FASTA parsing
    fasta_text = """>seq1 description1
ATGCCC
GGGTTT
>seq2
AAATTTCCC"""
    
    sequences = parse_fasta(fasta_text)
    assert len(sequences) == 2
    assert sequences[0]["id"] == "seq1"
    assert sequences[0]["description"] == "description1"
    assert sequences[0]["sequence"] == "ATGCCCGGGTTT"
    assert sequences[1]["id"] == "seq2"
    assert sequences[1]["sequence"] == "AAATTTCCC"
    
    # Test CSV parsing
    csv_text = """id,sequence,description
seq1,ATGCCC,test sequence
seq2,GGGTTT,another sequence"""
    
    sequences = parse_csv_tsv(csv_text, ",")
    assert len(sequences) == 2
    assert sequences[0]["id"] == "seq1"
    assert sequences[0]["sequence"] == "ATGCCC"
    assert sequences[0]["description"] == "test sequence"


def test_codon_table_constants():
    """Test that codon table constants are properly defined."""
    from server.constants import CODON_TABLE, ALL_64_CODONS
    
    # Should have all 64 codons
    assert len(CODON_TABLE) == 64
    assert len(ALL_64_CODONS) == 64
    
    # Test some known translations
    assert CODON_TABLE["ATG"] == "M"  # Start codon
    assert CODON_TABLE["TAA"] == "*"  # Stop codon
    assert CODON_TABLE["TTT"] == "F"  # Phenylalanine
    
    # All codons should be in the table
    for codon in ALL_64_CODONS:
        assert codon in CODON_TABLE
        assert len(codon) == 3
        assert all(base in "ATCG" for base in codon)


def test_pydantic_models():
    """Test Pydantic model definitions."""
    from visualizer.server import SequenceInput, BatchInput, ProteinInput
    
    # Test SequenceInput
    seq_input = SequenceInput(sequence="ATGCCC")
    assert seq_input.sequence == "ATGCCC"
    
    # Test ProteinInput
    protein_input = ProteinInput(protein="MALT", n_variants=2)
    assert protein_input.protein == "MALT"
    assert protein_input.n_variants == 2
    
    # Test BatchInput
    batch_input = BatchInput(
        format="json",
        sequences=[{"id": "seq1", "sequence": "ATGCCC"}]
    )
    assert batch_input.format == "json"
    assert len(batch_input.sequences) == 1


def test_basic_app_structure():
    """Test that the app has the expected endpoints defined."""
    from visualizer.server import app
    
    # Get all route paths
    routes = [route.path for route in app.routes if hasattr(route, 'path')]
    
    # Check for key API endpoints
    expected_endpoints = [
        "/api/encode",
        "/api/encode/batch",
        "/api/points",
        "/api/edges",
        "/api/metadata",
        "/api/visualization",
        "/api/synonymous_variants"
    ]
    
    for endpoint in expected_endpoints:
        assert endpoint in routes, f"Missing endpoint: {endpoint}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])