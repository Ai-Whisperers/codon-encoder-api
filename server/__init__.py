"""
Codon Inference Server

Standalone inference module for the Codon Encoder model.
The visualizer consumes this for served inference.
"""

from .inference_codon import (
    CodonEncoder,
    CODON_TABLE,
    BASE_TO_IDX,
    codon_to_onehot,
    extract_codons,
    compute_hyperbolic_radius,
    compute_depth_level,
    encode_custom_sequence,
)

__all__ = [
    "CodonEncoder",
    "CODON_TABLE",
    "BASE_TO_IDX",
    "codon_to_onehot",
    "extract_codons",
    "compute_hyperbolic_radius",
    "compute_depth_level",
    "encode_custom_sequence",
]
