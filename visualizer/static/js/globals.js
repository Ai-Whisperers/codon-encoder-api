/**
 * Global State and Constants
 */

// Global State
export const state = {
    currentEncodedSequence: [],
    currentEdges: [],
    projectionMode: 'poincare',
    serverConfig: {},
    angularVarianceData: null,
    synonymousVariants: [],
    overlayTrajectoryGroup: null,
    loadedSequences: [],
    visualizationData: null,
    actbVariants: [],
    varianceData: null
};

// Default colors (overwritten by server config when loaded)
export let DEPTH_COLORS = [
    0xff0000, 0xff4400, 0xff8800, 0xffcc00, 0xffff00,
    0x88ff00, 0x00ff88, 0x00ffff, 0x0088ff, 0x0000ff
];

export let AA_COLORS = {
    'A': 0x8B8B00, 'V': 0x8B8B00, 'L': 0x8B8B00, 'I': 0x8B8B00,
    'M': 0x8B8B00, 'F': 0x8B8B00, 'W': 0x8B8B00, 'P': 0x8B8B00,
    'S': 0x00CED1, 'T': 0x00CED1, 'N': 0x00CED1, 'Q': 0x00CED1,
    'Y': 0x00CED1, 'C': 0x00CED1, 'G': 0x00CED1,
    'K': 0x0000FF, 'R': 0x0000FF, 'H': 0x0000FF,
    'D': 0xFF0000, 'E': 0xFF0000,
    '*': 0x808080
};

// Parse hex color string to number
export function parseColorHex(hex) {
    if (typeof hex === 'number') return hex;
    return parseInt(hex.replace('#', ''), 16);
}

// Update colors from server config
export function applyServerColors(config) {
    if (config.depth_colors) {
        DEPTH_COLORS = config.depth_colors.map(parseColorHex);
    }
    if (config.aa_colors) {
        AA_COLORS = {};
        for (const [aa, hex] of Object.entries(config.aa_colors)) {
            AA_COLORS[aa] = parseColorHex(hex);
        }
    }
}

// =============================================================================
// RESEARCH DATA TABLES
// =============================================================================

// Kyte-Doolittle Hydrophobicity Scale (-4.5 to +4.5)
// Positive = hydrophobic, Negative = hydrophilic
export const HYDROPHOBICITY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
    '*': 0
};

// Amino Acid Molecular Weights (Daltons)
export const MOLECULAR_WEIGHT = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
    '*': 0
};

// Chemical Classification
export const CHEMICAL_CLASS = {
    // Aliphatic (nonpolar, aliphatic R groups)
    'G': 'aliphatic', 'A': 'aliphatic', 'V': 'aliphatic',
    'L': 'aliphatic', 'I': 'aliphatic', 'P': 'aliphatic',
    // Aromatic
    'F': 'aromatic', 'Y': 'aromatic', 'W': 'aromatic',
    // Sulfur-containing
    'C': 'sulfur', 'M': 'sulfur',
    // Hydroxyl (polar, uncharged)
    'S': 'hydroxyl', 'T': 'hydroxyl',
    // Amide (polar, uncharged)
    'N': 'amide', 'Q': 'amide',
    // Basic (positively charged)
    'K': 'basic', 'R': 'basic', 'H': 'basic',
    // Acidic (negatively charged)
    'D': 'acidic', 'E': 'acidic',
    // Stop
    '*': 'stop'
};

// Chemical Class Colors
export const CHEMICAL_CLASS_COLORS = {
    'aliphatic': 0x4a90d9,  // Blue
    'aromatic': 0x9b59b6,   // Purple
    'sulfur': 0xf1c40f,     // Yellow
    'hydroxyl': 0x2ecc71,   // Green
    'amide': 0x1abc9c,      // Teal
    'basic': 0x3498db,      // Light Blue
    'acidic': 0xe74c3c,     // Red
    'stop': 0x7f8c8d        // Gray
};

// Codon Degeneracy (number of codons encoding each amino acid)
export const DEGENERACY = {
    'M': 1, 'W': 1,                           // 1-fold
    'C': 2, 'D': 2, 'E': 2, 'F': 2, 'H': 2,   // 2-fold
    'K': 2, 'N': 2, 'Q': 2, 'Y': 2,
    'I': 3, '*': 3,                            // 3-fold
    'A': 4, 'G': 4, 'P': 4, 'T': 4, 'V': 4,   // 4-fold
    'L': 6, 'R': 6, 'S': 6                     // 6-fold
};

// Degeneracy Colors (1-fold rare, 6-fold common)
export const DEGENERACY_COLORS = {
    1: 0xff0000,  // Red - unique codon
    2: 0xff8800,  // Orange
    3: 0xffff00,  // Yellow
    4: 0x88ff00,  // Light green
    6: 0x00ff00   // Green - most redundant
};

// Wobble Base Colors (3rd position)
export const WOBBLE_COLORS = {
    'A': 0x00ff00,  // Green
    'T': 0xff0000,  // Red
    'G': 0xffff00,  // Yellow
    'C': 0x0088ff   // Blue
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Calculate GC content of a codon (0-100%)
export function getGCContent(codon) {
    if (!codon || codon.length !== 3) return 0;
    const gc = (codon.match(/[GC]/gi) || []).length;
    return (gc / 3) * 100;
}

// Get wobble base (3rd position)
export function getWobbleBase(codon) {
    if (!codon || codon.length !== 3) return 'N';
    return codon[2].toUpperCase();
}

// Normalize value to 0-1 range
export function normalize(value, min, max) {
    return (value - min) / (max - min);
}

// =============================================================================
// CODON TABLE AND AMINO ACID MAPPING
// =============================================================================

export const CODON_TABLE = {
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
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
};

// Amino acid full names
export const AA_NAMES = {
    'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartate',
    'C': 'Cysteine', 'E': 'Glutamate', 'Q': 'Glutamine', 'G': 'Glycine',
    'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
    'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
    'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine',
    '*': 'Stop'
};

// Get all codons for an amino acid
export function getCodonsForAA(aa) {
    return Object.entries(CODON_TABLE)
        .filter(([_, a]) => a === aa)
        .map(([codon]) => codon);
}

// =============================================================================
// ORGANISM-SPECIFIC CODON USAGE (frequency per 1000 codons)
// =============================================================================

// E. coli K12 codon usage
export const CODON_USAGE_ECOLI = {
    'TTT': 22.0, 'TTC': 16.6, 'TTA': 13.8, 'TTG': 13.7,
    'TCT': 8.6, 'TCC': 8.8, 'TCA': 7.6, 'TCG': 8.9,
    'TAT': 16.3, 'TAC': 12.2, 'TAA': 2.0, 'TAG': 0.3,
    'TGT': 5.2, 'TGC': 6.5, 'TGA': 1.0, 'TGG': 15.3,
    'CTT': 11.0, 'CTC': 11.0, 'CTA': 3.9, 'CTG': 52.6,
    'CCT': 7.2, 'CCC': 5.5, 'CCA': 8.4, 'CCG': 23.1,
    'CAT': 12.9, 'CAC': 9.7, 'CAA': 15.3, 'CAG': 29.0,
    'CGT': 20.9, 'CGC': 21.9, 'CGA': 3.7, 'CGG': 5.7,
    'ATT': 30.1, 'ATC': 25.1, 'ATA': 4.5, 'ATG': 27.6,
    'ACT': 9.0, 'ACC': 23.0, 'ACA': 7.6, 'ACG': 14.4,
    'AAT': 18.3, 'AAC': 21.4, 'AAA': 33.6, 'AAG': 10.3,
    'AGT': 9.0, 'AGC': 16.1, 'AGA': 2.3, 'AGG': 1.4,
    'GTT': 18.4, 'GTC': 15.3, 'GTA': 10.9, 'GTG': 26.3,
    'GCT': 15.5, 'GCC': 25.5, 'GCA': 20.3, 'GCG': 33.7,
    'GAT': 32.2, 'GAC': 19.2, 'GAA': 39.4, 'GAG': 18.0,
    'GGT': 24.7, 'GGC': 29.6, 'GGA': 8.1, 'GGG': 11.3
};

// S. cerevisiae (yeast) codon usage
export const CODON_USAGE_YEAST = {
    'TTT': 26.1, 'TTC': 18.4, 'TTA': 26.2, 'TTG': 27.2,
    'TCT': 23.5, 'TCC': 14.2, 'TCA': 18.7, 'TCG': 8.6,
    'TAT': 18.8, 'TAC': 14.8, 'TAA': 1.1, 'TAG': 0.5,
    'TGT': 8.1, 'TGC': 4.8, 'TGA': 0.7, 'TGG': 10.4,
    'CTT': 12.3, 'CTC': 5.4, 'CTA': 13.4, 'CTG': 10.5,
    'CCT': 13.5, 'CCC': 6.8, 'CCA': 18.3, 'CCG': 5.3,
    'CAT': 13.6, 'CAC': 7.8, 'CAA': 27.3, 'CAG': 12.1,
    'CGT': 6.4, 'CGC': 2.6, 'CGA': 3.0, 'CGG': 1.7,
    'ATT': 30.1, 'ATC': 17.2, 'ATA': 17.8, 'ATG': 20.9,
    'ACT': 20.3, 'ACC': 12.7, 'ACA': 17.8, 'ACG': 8.0,
    'AAT': 35.7, 'AAC': 24.8, 'AAA': 41.9, 'AAG': 30.8,
    'AGT': 14.2, 'AGC': 9.8, 'AGA': 21.3, 'AGG': 9.2,
    'GTT': 22.1, 'GTC': 11.8, 'GTA': 11.8, 'GTG': 10.8,
    'GCT': 21.2, 'GCC': 12.6, 'GCA': 16.2, 'GCG': 6.2,
    'GAT': 37.6, 'GAC': 20.2, 'GAA': 45.6, 'GAG': 19.2,
    'GGT': 23.9, 'GGC': 9.8, 'GGA': 10.9, 'GGG': 6.0
};

// H. sapiens (human) codon usage
export const CODON_USAGE_HUMAN = {
    'TTT': 17.6, 'TTC': 20.3, 'TTA': 7.7, 'TTG': 12.9,
    'TCT': 15.2, 'TCC': 17.7, 'TCA': 12.2, 'TCG': 4.4,
    'TAT': 12.2, 'TAC': 15.3, 'TAA': 1.0, 'TAG': 0.8,
    'TGT': 10.6, 'TGC': 12.6, 'TGA': 1.6, 'TGG': 13.2,
    'CTT': 13.2, 'CTC': 19.6, 'CTA': 7.2, 'CTG': 39.6,
    'CCT': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,
    'CAT': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2,
    'CGT': 4.5, 'CGC': 10.4, 'CGA': 6.2, 'CGG': 11.4,
    'ATT': 16.0, 'ATC': 20.8, 'ATA': 7.5, 'ATG': 22.0,
    'ACT': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,
    'AAT': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9,
    'AGT': 12.1, 'AGC': 19.5, 'AGA': 12.2, 'AGG': 12.0,
    'GTT': 11.0, 'GTC': 14.5, 'GTA': 7.1, 'GTG': 28.1,
    'GCT': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,
    'GAT': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6,
    'GGT': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5
};

// M. musculus (mouse) codon usage
export const CODON_USAGE_MOUSE = {
    'TTT': 17.2, 'TTC': 21.8, 'TTA': 6.7, 'TTG': 12.6,
    'TCT': 16.1, 'TCC': 18.6, 'TCA': 11.7, 'TCG': 4.5,
    'TAT': 12.0, 'TAC': 16.0, 'TAA': 0.9, 'TAG': 0.7,
    'TGT': 10.5, 'TGC': 13.4, 'TGA': 1.5, 'TGG': 12.8,
    'CTT': 12.8, 'CTC': 19.9, 'CTA': 7.8, 'CTG': 40.3,
    'CCT': 18.0, 'CCC': 19.4, 'CCA': 17.0, 'CCG': 6.7,
    'CAT': 10.4, 'CAC': 15.0, 'CAA': 11.8, 'CAG': 34.5,
    'CGT': 4.7, 'CGC': 10.5, 'CGA': 6.4, 'CGG': 11.8,
    'ATT': 15.6, 'ATC': 22.5, 'ATA': 7.1, 'ATG': 22.4,
    'ACT': 13.3, 'ACC': 19.7, 'ACA': 15.5, 'ACG': 6.2,
    'AAT': 16.4, 'AAC': 20.4, 'AAA': 23.2, 'AAG': 33.6,
    'AGT': 11.9, 'AGC': 19.6, 'AGA': 11.5, 'AGG': 11.7,
    'GTT': 10.8, 'GTC': 15.2, 'GTA': 7.0, 'GTG': 29.0,
    'GCT': 19.2, 'GCC': 28.1, 'GCA': 15.3, 'GCG': 7.3,
    'GAT': 21.2, 'GAC': 26.0, 'GAA': 28.0, 'GAG': 40.8,
    'GGT': 11.0, 'GGC': 23.3, 'GGA': 16.3, 'GGG': 16.0
};

// Get codon usage table by organism
export function getCodonUsage(organism) {
    switch (organism) {
        case 'ecoli': return CODON_USAGE_ECOLI;
        case 'yeast': return CODON_USAGE_YEAST;
        case 'human': return CODON_USAGE_HUMAN;
        case 'mouse': return CODON_USAGE_MOUSE;
        default: return null;
    }
}

// =============================================================================
// CODON STATISTICS FUNCTIONS
// =============================================================================

// Calculate Relative Synonymous Codon Usage (RSCU)
// RSCU = observed frequency / expected frequency (if all synonymous codons used equally)
export function calculateRSCU(codon, usage) {
    if (!usage) return 1.0;
    const aa = CODON_TABLE[codon];
    const synonymousCodons = getCodonsForAA(aa);
    const avgFreq = synonymousCodons.reduce((sum, c) => sum + (usage[c] || 0), 0) / synonymousCodons.length;
    if (avgFreq === 0) return 1.0;
    return (usage[codon] || 0) / avgFreq;
}

// Calculate Codon Adaptation Index (CAI) for a sequence
// CAI = geometric mean of relative adaptiveness values
export function calculateCAI(codons, usage) {
    if (!usage || codons.length === 0) return 0;

    // Calculate max frequency for each amino acid
    const maxFreqs = {};
    for (const [codon, aa] of Object.entries(CODON_TABLE)) {
        if (!maxFreqs[aa] || usage[codon] > maxFreqs[aa]) {
            maxFreqs[aa] = usage[codon];
        }
    }

    let logSum = 0;
    let count = 0;
    for (const codon of codons) {
        const aa = CODON_TABLE[codon];
        if (aa && aa !== '*' && maxFreqs[aa] > 0) {
            const w = (usage[codon] || 0.5) / maxFreqs[aa];
            logSum += Math.log(Math.max(w, 0.01));
            count++;
        }
    }

    return count > 0 ? Math.exp(logSum / count) : 0;
}

// Calculate Effective Number of Codons (ENC)
// ENC ranges from 20 (extreme bias) to 61 (no bias)
export function calculateENC(codons) {
    if (codons.length === 0) return 61;

    // Group codons by amino acid
    const aaUsage = {};
    for (const codon of codons) {
        const aa = CODON_TABLE[codon];
        if (!aa || aa === '*') continue;
        if (!aaUsage[aa]) aaUsage[aa] = {};
        aaUsage[aa][codon] = (aaUsage[aa][codon] || 0) + 1;
    }

    // Calculate F values for each degeneracy class
    const fValues = { 2: [], 3: [], 4: [], 6: [] };

    for (const [aa, codonCounts] of Object.entries(aaUsage)) {
        const deg = DEGENERACY[aa];
        if (!deg || deg === 1) continue;

        const total = Object.values(codonCounts).reduce((a, b) => a + b, 0);
        if (total < 2) continue;

        let sumPiSq = 0;
        for (const count of Object.values(codonCounts)) {
            const pi = count / total;
            sumPiSq += pi * pi;
        }

        const F = (total * sumPiSq - 1) / (total - 1);
        if (fValues[deg]) fValues[deg].push(F);
    }

    // Calculate ENC
    let enc = 2; // Start with Met + Trp (1-fold)
    for (const [deg, values] of Object.entries(fValues)) {
        if (values.length > 0) {
            const avgF = values.reduce((a, b) => a + b, 0) / values.length;
            const numAAs = deg === '2' ? 9 : deg === '3' ? 1 : deg === '4' ? 5 : 3;
            enc += numAAs / Math.max(avgF, 0.01);
        }
    }

    return Math.min(61, Math.max(20, enc));
}

// =============================================================================
// MUTATION NETWORK HELPERS
// =============================================================================

// Check if two codons differ by exactly one nucleotide
export function isSingleMutation(codon1, codon2) {
    if (!codon1 || !codon2 || codon1.length !== 3 || codon2.length !== 3) return false;
    let diffs = 0;
    for (let i = 0; i < 3; i++) {
        if (codon1[i] !== codon2[i]) diffs++;
    }
    return diffs === 1;
}

// Get all codons reachable by single mutation
export function getSingleMutationNeighbors(codon) {
    const neighbors = [];
    const bases = ['A', 'T', 'G', 'C'];
    for (let pos = 0; pos < 3; pos++) {
        for (const base of bases) {
            if (base !== codon[pos]) {
                const newCodon = codon.substring(0, pos) + base + codon.substring(pos + 1);
                neighbors.push({
                    codon: newCodon,
                    position: pos,
                    from: codon[pos],
                    to: base,
                    aa: CODON_TABLE[newCodon]
                });
            }
        }
    }
    return neighbors;
}
