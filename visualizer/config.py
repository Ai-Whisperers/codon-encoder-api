"""
Configuration module for Codon Encoder Visualizer.
Change MODEL_PATH to evaluate different model variants.
"""
import os
from pathlib import Path
from typing import Optional

# =============================================================================
# MODEL CONFIGURATION - Change this path to evaluate different models
# =============================================================================
_default_model = Path(os.getenv('CODON_MODEL_PATH',
    Path(__file__).parent.parent / "server" / "model" / "codon_encoder.pt"))

# Mutable state holder for runtime configuration
class Config:
    """Runtime configuration that can be modified."""
    model_path: Path = _default_model

    @classmethod
    def set_model_path(cls, path: Path):
        cls.model_path = Path(path).resolve()

    @classmethod
    def get_model_path(cls) -> Path:
        return cls.model_path


# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================
VIS_CONFIG = {
    # Server
    "host": "127.0.0.1",
    "port": 8765,

    # 3D Projection (server-side, used for PCA fallback)
    "projection_method": "pca",  # pca, umap, tsne
    "projection_dim": 3,

    # Fiber/edge rendering
    "fiber_mode": "hierarchical",  # hierarchical, amino_acid, depth, none
    "fiber_threshold": 1.5,        # Distance threshold for hierarchical connections (16-dim embedding space)

    # Color scales (depth 0-9) - sent to frontend
    "depth_colors": [
        "#ff0000",  # d=0 (outer, most common)
        "#ff4400",
        "#ff8800",
        "#ffcc00",
        "#ffff00",
        "#88ff00",
        "#00ff88",
        "#00ffff",
        "#0088ff",
        "#0000ff",  # d=9 (inner, rarest)
    ],

    # Amino acid color scheme (chemistry-based) - sent to frontend
    "aa_colors": {
        # Hydrophobic
        "A": "#8B8B00", "V": "#8B8B00", "L": "#8B8B00", "I": "#8B8B00",
        "M": "#8B8B00", "F": "#8B8B00", "W": "#8B8B00", "P": "#8B8B00",
        # Polar
        "S": "#00CED1", "T": "#00CED1", "N": "#00CED1", "Q": "#00CED1",
        "Y": "#00CED1", "C": "#00CED1", "G": "#00CED1",
        # Positive
        "K": "#0000FF", "R": "#0000FF", "H": "#0000FF",
        # Negative
        "D": "#FF0000", "E": "#FF0000",
        # Stop
        "*": "#808080",
    },
}

# =============================================================================
# CODON CONSTANTS
# =============================================================================
BASES = ['T', 'C', 'A', 'G']
ALL_64_CODONS = [b1 + b2 + b3 for b1 in BASES for b2 in BASES for b3 in BASES]

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

BASE_TO_IDX = {'T': 0, 'C': 1, 'A': 2, 'G': 3}
