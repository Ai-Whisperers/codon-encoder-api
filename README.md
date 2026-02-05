# Codon Encoder API

**A FastAPI service for DNA sequence encoding and visualization using hyperbolic embeddings**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

## Overview

The Codon Encoder API provides state-of-the-art DNA sequence encoding using hierarchical hyperbolic embeddings. Transform DNA sequences into rich representations for visualization, analysis, and machine learning applications.

### Key Features

- 🧬 **DNA Sequence Encoding** - Convert DNA sequences to 16-dimensional hyperbolic embeddings
- 📊 **3D Visualization** - PCA/UMAP/t-SNE projections for interactive exploration
- 🔄 **Synonymous Variants** - Generate alternative DNA sequences for the same protein
- 🏷️ **Amino Acid Classification** - Cluster assignments with confidence scores
- 🌐 **RESTful API** - Modern FastAPI with OpenAPI documentation
- 🐳 **Docker Ready** - Containerized deployment with health checks

### Scientific Background

This API leverages hierarchical codon embeddings trained on large-scale genomic data to capture:

- **Genetic code redundancy** - Synonymous codons cluster by amino acid
- **Evolutionary relationships** - Related codons exhibit spatial proximity
- **Usage bias** - Frequent codons occupy central positions
- **Hierarchical structure** - Multi-level organization from nucleotides to proteins

## Quick Start

### Prerequisites

- Docker Engine
- Model file: `server/model/codon_encoder.pt` (contact us for access)

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/Ai-Whisperers/codon-encoder-api.git
cd codon-encoder-api

# Add model file (required)
mkdir -p server/model
cp /path/to/codon_encoder.pt server/model/

# Start service
docker compose up -d

# Verify
curl http://localhost:8765/api/metadata
```

### Option 2: Local Development

```bash
# Install dependencies
pip install .

# Set model path
export CODON_MODEL_PATH=./server/model/codon_encoder.pt

# Start server
cd visualizer
python run.py
```

## API Usage

### Encode DNA Sequence

```bash
curl -X POST "http://localhost:8765/api/encode" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "ATGGCTCTGTGG"}'
```

### Batch Processing

```python
import requests

# Multiple sequences
response = requests.post("http://localhost:8765/api/encode/batch", json={
    "format": "json",
    "sequences": [
        {"id": "seq1", "sequence": "ATGGCTCTG"},
        {"id": "seq2", "sequence": "TGGCTCTGA"}
    ]
})
```

### Generate Synonymous Variants

```bash
curl -X POST "http://localhost:8765/api/synonymous_variants" \
     -H "Content-Type: application/json" \
     -d '{"protein": "MALTV", "n_variants": 3}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/encode` | POST | Encode DNA sequence |
| `/api/encode/batch` | POST | Batch sequence encoding |
| `/api/synonymous_variants` | POST | Generate DNA variants |
| `/api/visualization` | GET | Complete dataset |
| `/api/points` | GET | All 64 codon points |
| `/api/edges` | GET | Fiber connections |
| `/api/metadata` | GET | Model information |
| `/api/codon/{codon}` | GET | Single codon details |
| `/api/cluster/{idx}` | GET | Amino acid cluster |

## Data Format

### Encoded Codon Response

```json
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
}
```

### Key Fields

- **projection**: 3D coordinates for visualization
- **depth**: Hierarchical level (0-9)
- **confidence**: Classification certainty
- **cluster_idx**: Amino acid group assignment
- **poincare_radius**: Hyperbolic distance metric

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/Ai-Whisperers/codon-encoder-api.git
cd codon-encoder-api

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=visualizer --cov=server

# Specific test
pytest tests/test_server.py::test_encode_endpoint
```

### Code Quality

```bash
# Linting
ruff check .
ruff check . --fix

# Formatting
black .
isort .

# Type checking
mypy .

# Security scan
bandit -r server/ visualizer/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Deployment

### Local Development

```bash
docker compose up -d
```

### Production with Cloudflare Tunnel

```bash
# Create tunnel
cloudflared tunnel create codon-api
cloudflared tunnel route dns codon-api codon.ai-whisperers.org

# Add token to environment
echo "CLOUDFLARE_TUNNEL_TOKEN=<token>" > .env

# Deploy with tunnel
docker compose --profile tunnel up -d
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODON_MODEL_PATH` | `/app/server/model/codon_encoder.pt` | Model file location |
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `8765` | Server port |

## Client Libraries

### Python

```python
# Install client
pip install httpx pydantic

# Use client
from contracts.python import CodonClient, encode_sequence

# Quick encoding
codons = encode_sequence("ATGGCTCTG")
for c in codons:
    print(f"{c.codon} -> {c.amino_acid} (depth={c.depth})")

# Full client
client = CodonClient()
result = client.encode("ATGGCTCTG")
```

### TypeScript

```typescript
import { CodonClient } from './contracts/typescript/client';

const client = new CodonClient();
const result = await client.encode("ATGGCTCTG");
```

## Architecture

```
codon-encoder-api/
├── server/              # Core inference logic
│   ├── inference_codon.py   # Main encoding functions
│   ├── constants.py         # Genetic code definitions
│   └── utils/              # Analysis utilities
├── visualizer/          # FastAPI web service
│   ├── server.py           # Main API endpoints
│   ├── model_loader.py     # Model management
│   └── config.py           # Configuration
├── contracts/           # Client libraries
│   ├── python/             # Python client
│   ├── typescript/         # TypeScript client
│   └── openapi/            # OpenAPI spec
├── tests/               # Test suite
└── docs/               # Documentation
```

## Model Details

- **Architecture**: Hierarchical hyperbolic embeddings
- **Input**: DNA sequences (any length, multiple of 3)
- **Output**: 16-dimensional codon embeddings
- **Projections**: PCA, UMAP, t-SNE for 3D visualization
- **Clusters**: 21 amino acid groups (20 + stop codon)

### What's Exposed vs Protected

**Public API** provides derived metrics only:
- 3D projections for visualization
- Cluster assignments and confidence scores
- Hierarchical depth levels
- Geometric properties (norms, distances)

**Protected** intellectual property:
- 16-dimensional raw embeddings
- Model weights and architecture
- Training data and procedures

## Performance

- **Latency**: ~10ms per sequence (100 codons)
- **Throughput**: 1000+ sequences/second
- **Memory**: ~2GB RAM for model
- **Batch size**: Up to 1000 sequences

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Note on Model Weights

While the API code is open source, the trained model weights (`codon_encoder.pt`) are proprietary intellectual property of AI Whisperers. Contact us at api@ai-whisperers.org for research access.

## Support

- 📧 Email: api@ai-whisperers.org
- 🐛 Issues: [GitHub Issues](https://github.com/Ai-Whisperers/codon-encoder-api/issues)
- 📚 Documentation: [API Documentation](./docs/)
- 🌐 Demo: [codon.ai-whisperers.org](https://codon.ai-whisperers.org)

## Citation

If you use this API in your research, please cite:

```bibtex
@software{codon_encoder_api,
  title={Codon Encoder API: Hierarchical Hyperbolic Embeddings for DNA Sequences},
  author={AI Whisperers},
  year={2025},
  url={https://github.com/Ai-Whisperers/codon-encoder-api}
}
```

---

**Made with 🧬 by [AI Whisperers](https://ai-whisperers.org)**