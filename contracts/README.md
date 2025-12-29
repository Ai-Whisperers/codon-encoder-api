# Codon Inference API Contracts

**Doc-Type:** API Contract Documentation · Version 1.0 · Updated 2025-12-29 · Author AI Whisperers

Open-source API contracts for consuming the Codon Encoder inference service.

---

## Overview

**purpose** - Define public contracts for the Codon Inference API
**endpoint** - `https://codon.ai-whisperers.org/api`
**security** - Model weights and raw embeddings are NOT exposed

This package provides type-safe client implementations and OpenAPI specifications for consuming the Codon Encoder inference service. The API enables:

- DNA sequence encoding to codon trajectories
- 3D projections for visualization
- Cluster (amino acid) assignments
- Hierarchical depth analysis
- Synonymous variant generation

---

## What's Exposed vs Protected

### Exposed (Public API)

| Data | Description | Use Case |
|------|-------------|----------|
| 3D Projections | PCA/UMAP/t-SNE coordinates | Visualization |
| Cluster Index | Amino acid assignment (0-20) | Classification |
| Confidence | 1/(1+distance_to_nearest) | Quality metric |
| Margin | Second_dist - First_dist | Decision boundary |
| Depth | Hierarchical level (0-9) | Hierarchical position |
| Poincaré Radius | Hyperbolic distance | Geometric visualization |
| Embedding Norm | `||embedding||` scalar | Magnitude indicator |

### Protected (Not Exposed)

| Data | Reason |
|------|--------|
| 16-dim Embeddings | Model intellectual property |
| Cluster Centers | Model weights |
| Encoder Weights | Model architecture |
| Distance Matrices | Derivable to embeddings |

---

## Directory Structure

```
contracts/
├── README.md              # This file
├── python/
│   ├── __init__.py        # Package exports
│   ├── models.py          # Pydantic response models
│   ├── client.py          # HTTP client implementation
│   └── examples.py        # Usage examples
├── typescript/
│   ├── types.ts           # TypeScript interfaces
│   ├── client.ts          # Fetch-based client
│   └── examples.ts        # Usage examples
└── openapi/
    └── openapi.yaml       # OpenAPI 3.1 specification
```

---

## Python Client

### Installation

```bash
pip install httpx pydantic
```

### Quick Start

```python
from codon_client import CodonClient, encode_sequence

# Quick encoding
codons = encode_sequence("ATGGCTCTGTGG")
for c in codons:
    print(f"{c.codon} -> {c.amino_acid} (depth={c.depth})")

# Full client
client = CodonClient()
result = client.encode("ATGGCTCTGTGG")
metadata = client.get_metadata()
variants = client.get_synonymous_variants("MDDII", n_variants=3)
```

### Available Methods

| Method | Description |
|--------|-------------|
| `encode(sequence)` | DNA → codon trajectory |
| `get_visualization()` | Full 64-point dataset |
| `get_points()` | All codon points |
| `get_edges(mode)` | Fiber connections |
| `get_metadata()` | Model info (no weights) |
| `get_codon(codon)` | Single codon lookup |
| `get_cluster(idx)` | Amino acid group |
| `get_depth(level)` | Codons at depth level |
| `get_synonymous_variants(protein)` | DNA alternatives |
| `get_angular_variance()` | Separation statistics |

---

## TypeScript Client

### Installation

```bash
npm install # (copy types.ts and client.ts to your project)
```

### Quick Start

```typescript
import { CodonClient, encodeSequence } from './client';

// Quick encoding
const codons = await encodeSequence("ATGGCTCTGTGG");
codons.forEach(c => {
  console.log(`${c.codon} -> ${c.amino_acid} (depth=${c.depth})`);
});

// Full client
const client = new CodonClient();
const result = await client.encode("ATGGCTCTGTGG");
const metadata = await client.getMetadata();
const variants = await client.getSynonymousVariants("MDDII", 3);
```

### React Hook Example

```tsx
import { useState, useMemo, useCallback } from 'react';
import { CodonClient, EncodedCodon } from 'codon-client';

export function useCodonEncoder() {
  const [result, setResult] = useState<EncodedCodon[]>([]);
  const [loading, setLoading] = useState(false);
  const client = useMemo(() => new CodonClient(), []);

  const encode = useCallback(async (sequence: string) => {
    setLoading(true);
    const data = await client.encode(sequence);
    setResult(data);
    setLoading(false);
  }, [client]);

  return { result, loading, encode };
}
```

---

## API Endpoints

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/encode` | Encode DNA sequence |
| POST | `/api/synonymous_variants` | Generate protein variants |

### Visualization

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/visualization` | Complete dataset |
| GET | `/api/points` | All 64 codon points |
| GET | `/api/edges?mode=` | Fiber connections |
| GET | `/api/metadata` | Model info |

### Lookup

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/codon/{codon}` | Single codon |
| GET | `/api/cluster/{idx}` | Amino acid group |
| GET | `/api/depth/{level}` | Depth level |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/angular_variance` | Separation stats |
| GET | `/api/reference/actb` | ACTB reference gene |

---

## Response Examples

### POST /api/encode

Request:
```json
{"sequence": "ATGGCT"}
```

Response:
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
    "cluster_idx": 10,
    "confidence": 0.89,
    "margin": 0.45
  },
  ...
]
```

### GET /api/metadata

```json
{
  "version": "v5.11.3",
  "structure_score": -0.74,
  "cluster_accuracy": 0.95,
  "embed_dim": 16,
  "num_clusters": 21,
  "projection_method": "PCA"
}
```

---

## Rate Limits

| Tier | Limit |
|------|-------|
| Anonymous | 100 req/min |
| API Key | 1000 req/min |

Request an API key for higher limits: api@ai-whisperers.org

---

## Local Development

Start the visualizer server locally:

```bash
cd visualizer
python run.py
```

Connect clients to local server:

```python
client = CodonClient(base_url="http://localhost:8000")
```

```typescript
const client = new CodonClient({ baseUrl: "http://localhost:8000" });
```

---

## OpenAPI Specification

The complete OpenAPI 3.1 specification is available at:
- Local: `openapi/openapi.yaml`
- Production: `https://codon.ai-whisperers.org/api/openapi.json`

Generate clients for other languages:

```bash
# Using openapi-generator
openapi-generator generate -i openapi/openapi.yaml -g go -o ./go-client
openapi-generator generate -i openapi/openapi.yaml -g rust -o ./rust-client
```

---

## License

MIT License - See repository root for details.

**Note**: These contracts are open-source. The trained model weights are proprietary and not included.
