# Codon Encoder Visualizer - Development Status

**Doc-Type:** Technical Status - Version 1.2 - Updated 2025-12-28

## Status: PRODUCTION READY

All components are properly connected.

## Architecture

```
Backend (Python)                    Frontend (Three.js)
-----------------                   -------------------
config.py                           index.html
  - VIS_CONFIG with colors            - applyServerColors()
  - fiber_threshold                   - loadEdges()
                                      - renderFibers()
model_loader.py                       - projectPoint()
  - CodonPoint dataclass              - Depth rings
  - compute_edges()                   - Multiple projection modes
  - compute_angular_variance()

server.py
  - /api/visualization
  - /api/edges
  - /api/encode
  - /api/angular_variance
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/visualization` | GET | Full visualization payload |
| `/api/points` | GET | Codon points only |
| `/api/edges?mode=X` | GET | Fiber connections |
| `/api/metadata` | GET | Model metadata |
| `/api/encode` | POST | Encode DNA sequence |
| `/api/codon/{codon}` | GET | Single codon info |
| `/api/depth/{val}` | GET | Codons at depth level |
| `/api/angular_variance` | GET | Angular stats |
| `/api/synonymous_variants` | POST | Generate DNA variants |
| `/api/reference/actb` | GET | ACTB reference |

## Projection Modes

- Poincare disk
- Hemisphere
- PCA 3D
