"""
Consolidated constants for Codon Encoder API.

This module centralizes all constants used across the project to avoid
duplication and ensure consistency.
"""

from typing import Final

# =============================================================================
# NUCLEOTIDE BASES
# =============================================================================
BASES: Final[list[str]] = ["T", "C", "A", "G"]
BASE_TO_IDX: Final[dict[str, int]] = {"T": 0, "C": 1, "A": 2, "G": 3}
IDX_TO_BASE: Final[dict[int, str]] = {0: "T", 1: "C", 2: "A", 3: "G"}

# =============================================================================
# ALL 64 CODONS (standard genetic code order)
# =============================================================================
ALL_64_CODONS: Final[list[str]] = [b1 + b2 + b3 for b1 in BASES for b2 in BASES for b3 in BASES]

# =============================================================================
# STANDARD GENETIC CODE (DNA -> Amino Acid)
# =============================================================================
CODON_TABLE: Final[dict[str, str]] = {
    # Phenylalanine (F)
    "TTT": "F",
    "TTC": "F",
    # Leucine (L)
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    # Serine (S)
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "AGT": "S",
    "AGC": "S",
    # Tyrosine (Y)
    "TAT": "Y",
    "TAC": "Y",
    # Stop codons (*)
    "TAA": "*",
    "TAG": "*",
    "TGA": "*",
    # Cysteine (C)
    "TGT": "C",
    "TGC": "C",
    # Tryptophan (W)
    "TGG": "W",
    # Proline (P)
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    # Histidine (H)
    "CAT": "H",
    "CAC": "H",
    # Glutamine (Q)
    "CAA": "Q",
    "CAG": "Q",
    # Arginine (R)
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGA": "R",
    "AGG": "R",
    # Isoleucine (I)
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    # Methionine (M) - Start codon
    "ATG": "M",
    # Threonine (T)
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    # Asparagine (N)
    "AAT": "N",
    "AAC": "N",
    # Lysine (K)
    "AAA": "K",
    "AAG": "K",
    # Valine (V)
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    # Alanine (A)
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    # Aspartic acid (D)
    "GAT": "D",
    "GAC": "D",
    # Glutamic acid (E)
    "GAA": "E",
    "GAG": "E",
    # Glycine (G)
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Reverse mapping: Amino Acid -> list of codons
AA_TO_CODONS: Final[dict[str, list[str]]] = {}
for codon, aa in CODON_TABLE.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)

# All amino acids (including stop)
AMINO_ACIDS: Final[list[str]] = sorted(set(CODON_TABLE.values()))

# =============================================================================
# MODEL ARCHITECTURE CONSTANTS
# =============================================================================
INPUT_DIM: Final[int] = 12  # 4 bases × 3 positions (one-hot)
HIDDEN_DIM: Final[int] = 32  # Hidden layer dimension
EMBEDDING_DIM: Final[int] = 16  # Output embedding dimension
NUM_CLUSTERS: Final[int] = 21  # 20 amino acids + 1 stop codon

# =============================================================================
# HYPERBOLIC GEOMETRY CONSTANTS
# =============================================================================
MAX_DEPTH: Final[int] = 9  # Maximum depth level (0-9)
HIERARCHY_FACTOR: Final[int] = 3  # Branching factor for hierarchy
POINCARE_OUTER_RADIUS: Final[float] = 0.9  # Radius at depth 0 (outer)
POINCARE_CENTER_RADIUS: Final[float] = 0.1  # Radius at depth 9 (center)

# =============================================================================
# SEQUENCE VALIDATION LIMITS
# =============================================================================
MIN_SEQUENCE_LENGTH: Final[int] = 3  # Minimum DNA sequence length
MAX_SEQUENCE_LENGTH: Final[int] = 100_000  # Maximum DNA sequence length
MAX_BATCH_SIZE: Final[int] = 1000  # Maximum sequences per batch

# =============================================================================
# VISUALIZATION COLORS
# =============================================================================
DEPTH_COLORS: Final[list[str]] = [
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
]

# Amino acid color scheme (chemistry-based)
AA_COLORS: Final[dict[str, str]] = {
    # Hydrophobic (olive/brown)
    "A": "#8B8B00",
    "V": "#8B8B00",
    "L": "#8B8B00",
    "I": "#8B8B00",
    "M": "#8B8B00",
    "F": "#8B8B00",
    "W": "#8B8B00",
    "P": "#8B8B00",
    # Polar (cyan)
    "S": "#00CED1",
    "T": "#00CED1",
    "N": "#00CED1",
    "Q": "#00CED1",
    "Y": "#00CED1",
    "C": "#00CED1",
    "G": "#00CED1",
    # Positive charge (blue)
    "K": "#0000FF",
    "R": "#0000FF",
    "H": "#0000FF",
    # Negative charge (red)
    "D": "#FF0000",
    "E": "#FF0000",
    # Stop codon (gray)
    "*": "#808080",
}

# =============================================================================
# DEFAULT SERVER CONFIGURATION
# =============================================================================
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8765
DEFAULT_PROJECTION_METHOD: Final[str] = "pca"
DEFAULT_FIBER_MODE: Final[str] = "hierarchical"
DEFAULT_FIBER_THRESHOLD: Final[float] = 1.5
