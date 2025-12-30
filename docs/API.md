# Codon Encoder API Reference

## Base URL

```
https://codon.ai-whisperers.org/api
```

Local development:
```
http://localhost:8765/api
```

---

## Endpoints

### GET /api/metadata

Get model metadata and version information.

**Response:**
```json
{
  "model_path": "/app/server/model/codon_encoder.pt",
  "version": "3-adic",
  "structure_score": -0.832,
  "cluster_accuracy": 0.969,
  "n_codons": 64,
  "embed_dim": 16
}
```

---

### GET /api/visualization

Get complete visualization data including all points, edges, and configuration.

**Response:**
```json
{
  "points": [...],
  "edges": [...],
  "cluster_centers_3d": [...],
  "metadata": {...},
  "config": {...}
}
```

---

### GET /api/points

Get all 64 codon points with inference results.

**Response:**
```json
[
  {
    "codon": "ATG",
    "amino_acid": "M",
    "position": 9858,
    "depth": 1,
    "projection": [0.123, -0.456, 0.789],
    "embedding_norm": 2.34,
    "poincare_radius": 0.811,
    "cluster_idx": 19,
    "confidence": 0.89,
    "margin": 0.45
  },
  ...
]
```

---

### GET /api/edges

Get fiber/edge connections between codons.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `hierarchical` | Connection mode |

**Mode options:**
- `hierarchical` - Same depth, embedding distance < threshold
- `amino_acid` - Synonymous codons (same translation)
- `depth` - Adjacent depth levels
- `none` - No edges

**Response:**
```json
[
  {"source": 0, "target": 5, "weight": 0.85, "type": "hierarchical"},
  ...
]
```

---

### POST /api/encode

Encode a DNA sequence into codon trajectory.

**Request:**
```json
{
  "sequence": "ATGGCTCTGTGG"
}
```

**Response:**
```json
[
  {
    "seq_position": 0,
    "codon": "ATG",
    "amino_acid": "M",
    "ref_idx": 14,
    "projection": [0.123, -0.456, 0.789],
    "depth": 1,
    "poincare_radius": 0.811,
    "embedding_norm": 2.34,
    "cluster_idx": 19,
    "confidence": 0.89,
    "margin": 0.45
  },
  ...
]
```

---

### POST /api/encode/batch

Encode multiple sequences in batch (CSV, JSON, or FASTA format).

**Request (JSON array):**
```json
{
  "sequences": [
    {"id": "seq1", "sequence": "ATGGCTCTGTGG"},
    {"id": "seq2", "sequence": "ATGAAAGGG"}
  ],
  "format": "json"
}
```

**Request (FASTA):**
```json
{
  "sequences": ">seq1\nATGGCTCTGTGG\n>seq2\nATGAAAGGG",
  "format": "fasta"
}
```

**Request (CSV):**
```json
{
  "sequences": "id,sequence\nseq1,ATGGCTCTGTGG\nseq2,ATGAAAGGG",
  "format": "csv"
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "seq1",
      "sequence": "ATGGCTCTGTGG",
      "encoded": [...],
      "protein": "MAL",
      "stats": {
        "length": 4,
        "mean_depth": 1.5,
        "mean_confidence": 0.87
      }
    },
    ...
  ],
  "summary": {
    "total_sequences": 2,
    "total_codons": 7,
    "processing_time_ms": 15
  }
}
```

---

### GET /api/codon/{codon}

Get detailed info for a specific codon.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `codon` | string | Three-letter codon (e.g., ATG) |

**Response:**
```json
{
  "codon": "ATG",
  "amino_acid": "M",
  "position": 9858,
  "depth": 1,
  "projection": [0.123, -0.456, 0.789],
  "poincare_radius": 0.811,
  "cluster_idx": 19,
  "confidence": 0.89,
  "margin": 0.45
}
```

---

### GET /api/cluster/{idx}

Get all codons in an amino acid cluster.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `idx` | integer | Cluster index (0-20) |

**Response:**
```json
{
  "cluster_idx": 19,
  "amino_acid": "M",
  "codons": [...]
}
```

---

### GET /api/depth/{level}

Get all codons at a specific depth level.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | integer | Depth level (0-9) |

**Response:**
```json
{
  "depth": 0,
  "count": 24,
  "codons": [...]
}
```

---

### GET /api/angular_variance

Get angular variance statistics by depth level.

**Response:**
```json
{
  "by_depth": {
    "0": {
      "depth": 0,
      "n_codons": 24,
      "n_amino_acids": 6,
      "intra_aa_variance": 0.123,
      "inter_aa_variance": 0.456,
      "ratio": 3.7,
      "amino_acids": ["L", "R", "S", "A", "G", "P"]
    },
    ...
  },
  "summary": {
    "mean_intra_variance": 0.15,
    "mean_inter_variance": 0.42,
    "mean_ratio": 2.8
  }
}
```

---

### POST /api/synonymous_variants

Generate synonymous DNA variants for a protein sequence.

**Request:**
```json
{
  "protein": "MDDII",
  "n_variants": 3
}
```

**Response:**
```json
[
  {
    "variant_idx": 0,
    "strategy": "random",
    "dna_sequence": "ATGGATGATATCATC",
    "protein": "MDDII",
    "encoded": [...],
    "mean_depth": 1.2,
    "depth_std": 0.8
  },
  ...
]
```

---

### GET /api/reference/actb

Get ACTB (beta-actin) reference with synonymous variants.

**Response:**
```json
{
  "gene": "ACTB",
  "description": "Beta-actin (first 60 AA)",
  "protein": "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQS",
  "variants": [...]
}
```

---

## Error Responses

All errors return JSON with `detail` field:

```json
{
  "detail": "Error message here"
}
```

**Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 404 | Not found |
| 500 | Internal server error |
| 503 | Model not loaded |
