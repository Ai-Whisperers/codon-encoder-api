# Codon Encoder API - Deployment Guide

## Prerequisites

- Docker Engine (via WSL2 or native Linux)
- docker-compose v2+
- Model file: `server/model/codon_encoder.pt` (not included in repo)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Ai-Whisperers/codon-encoder-api.git
cd codon-encoder-api

# 2. Add your model file
cp /path/to/codon_encoder.pt server/model/

# 3. Build and start
docker compose up -d --build

# 4. Verify
curl http://localhost:8765/api/metadata
```

## Deployment Options

### Local Development

```bash
docker compose up -d
# API available at http://localhost:8765
```

### Cloudflare Tunnel

```bash
# Option 1: Quick tunnel (temporary URL)
docker compose up -d
cloudflared tunnel --url http://localhost:8765

# Option 2: Named tunnel (persistent)
cloudflared tunnel create codon-api
cloudflared tunnel route dns codon-api codon.ai-whisperers.org
CLOUDFLARE_TUNNEL_TOKEN=<token> docker compose --profile tunnel up -d
```

### Render.com

1. Connect GitHub repository
2. Select "Docker" environment
3. Set build command: `docker build -t codon-api .`
4. Set start command: `docker run -p 8765:8765 -v /path/to/model:/app/server/model:ro codon-api`
5. Add model file via Render Disks

### Vercel (Serverless)

Not recommended - requires model file access and persistent state.

### Fly.io

```bash
# Install flyctl
fly launch --dockerfile Dockerfile
fly secrets set CODON_MODEL_PATH=/app/server/model/codon_encoder.pt
fly volumes create model_data --size 1
fly deploy
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODON_MODEL_PATH` | `/app/server/model/codon_encoder.pt` | Path to model file |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8765` | Server port |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metadata` | GET | Model info |
| `/api/visualization` | GET | Full visualization data |
| `/api/encode` | POST | Encode DNA sequence |
| `/api/points` | GET | All 64 codon points |
| `/api/edges` | GET | Fiber connections |

## Health Check

```bash
curl http://localhost:8765/api/metadata
```

## Logs

```bash
docker compose logs -f api
```

## Security Notes

- Model file (`.pt`) is mounted read-only
- API runs as non-root user
- No raw embeddings exposed via API
- Only derived metrics (projections, clusters) available
