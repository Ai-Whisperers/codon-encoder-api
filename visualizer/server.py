"""
FastAPI server for Codon Encoder Visualizer.
Serves visualization data and handles live inference.
"""
import sys
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from contextlib import asynccontextmanager

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from config import Config, VIS_CONFIG
from model_loader import ModelLoader, VisualizationData


# =============================================================================
# APP STATE
# =============================================================================
class AppState:
    """Application state container."""
    loader: Optional[ModelLoader] = None
    vis_data: Optional[VisualizationData] = None

    @classmethod
    def initialize(cls, model_path: Optional[Path] = None):
        """Initialize model loader with optional path override."""
        if model_path:
            Config.set_model_path(model_path)

        path = Config.get_model_path()
        print(f"Loading model from: {path}")

        cls.loader = ModelLoader(path).load()
        cls.vis_data = cls.loader.get_visualization_data()
        print(f"Model loaded. {len(cls.vis_data.points)} codons, {len(cls.vis_data.edges)} edges")

    @classmethod
    def reload(cls, model_path: Path):
        """Reload with a new model."""
        Config.set_model_path(model_path)
        cls.loader = ModelLoader(model_path).load()
        cls.vis_data = cls.loader.get_visualization_data()


# =============================================================================
# LIFESPAN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup: Load model
    AppState.initialize()
    yield
    # Shutdown: Cleanup (if needed)
    AppState.loader = None
    AppState.vis_data = None


# =============================================================================
# APP SETUP
# =============================================================================
app = FastAPI(
    title="Codon Encoder Visualizer",
    description="Interactive visualization of hyperbolic codon embeddings",
    version="1.0.0",
    lifespan=lifespan
)


class SequenceInput(BaseModel):
    """DNA sequence for encoding."""
    sequence: str


class ModelPathInput(BaseModel):
    """New model path to load."""
    path: str


# =============================================================================
# STATIC FILES - Mount after API routes to avoid conflicts
# =============================================================================
static_path = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve main visualization page."""
    return FileResponse(static_path / "index.html")


# Mount static files for assets (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/api/visualization")
async def get_visualization():
    """Get complete visualization data."""
    if AppState.vis_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return asdict(AppState.vis_data)


@app.get("/api/points")
async def get_points():
    """Get codon points only."""
    if AppState.vis_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return AppState.vis_data.points


@app.get("/api/edges")
async def get_edges(mode: str = Query("hierarchical")):
    """Get edges with specified fiber mode."""
    if AppState.loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    edges = AppState.loader.compute_edges(mode)
    return edges


@app.get("/api/metadata")
async def get_metadata():
    """Get model metadata."""
    if AppState.vis_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return AppState.vis_data.metadata


@app.post("/api/encode")
async def encode_sequence(input: SequenceInput):
    """Encode DNA sequence and return visualization data."""
    if AppState.loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not input.sequence:
        raise HTTPException(status_code=400, detail="Empty sequence")

    # Clean and validate
    clean = ''.join(c.upper() for c in input.sequence if c.upper() in 'ATCG')
    if len(clean) < 3:
        raise HTTPException(status_code=400, detail="Sequence too short")

    encoded = AppState.loader.encode_sequence(clean)
    return encoded


@app.post("/api/load_model")
async def load_new_model(input: ModelPathInput):
    """Load a different model file."""
    model_path = Path(input.path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {input.path}")

    try:
        AppState.reload(model_path)
        return {
            "status": "ok",
            "model": str(model_path),
            "n_codons": len(AppState.vis_data.points),
            "metadata": AppState.vis_data.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/api/codon/{codon}")
async def get_codon_info(codon: str):
    """Get detailed info for a specific codon."""
    if AppState.vis_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    codon = codon.upper()
    for point in AppState.vis_data.points:
        if point['codon'] == codon:
            return point

    raise HTTPException(status_code=404, detail=f"Codon not found: {codon}")


@app.get("/api/cluster/{cluster_idx}")
async def get_cluster_codons(cluster_idx: int):
    """Get all codons in a cluster (amino acid group)."""
    if AppState.vis_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    codons = [p for p in AppState.vis_data.points if p['cluster_idx'] == cluster_idx]
    if not codons:
        raise HTTPException(status_code=404, detail=f"Cluster not found: {cluster_idx}")

    return {
        "cluster_idx": cluster_idx,
        "amino_acid": codons[0]['amino_acid'] if codons else '?',
        "codons": codons
    }


@app.get("/api/depth/{val}")
async def get_depth_layer(val: int):
    """Get all codons at a specific depth level."""
    if AppState.vis_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not 0 <= val <= 9:
        raise HTTPException(status_code=400, detail="Depth must be 0-9")

    codons = [p for p in AppState.vis_data.points if p['depth'] == val]
    return {
        "depth": val,
        "count": len(codons),
        "codons": codons
    }


@app.get("/api/angular_variance")
async def get_angular_variance():
    """Get intra-AA vs inter-AA angular variance at each depth level."""
    if AppState.loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return AppState.loader.compute_angular_variance()


class ProteinInput(BaseModel):
    """Protein sequence for synonymous recoding."""
    protein: str
    n_variants: int = 3


@app.post("/api/synonymous_variants")
async def get_synonymous_variants(input: ProteinInput):
    """Generate synonymous DNA variants for a protein sequence."""
    if AppState.loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not input.protein:
        raise HTTPException(status_code=400, detail="Empty protein sequence")

    # Clean protein sequence
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY*")
    clean = ''.join(c.upper() for c in input.protein if c.upper() in valid_aa)

    if len(clean) < 1:
        raise HTTPException(status_code=400, detail="No valid amino acids in sequence")

    variants = AppState.loader.generate_synonymous_variants(
        clean,
        n_variants=min(input.n_variants, 5)  # Cap at 5
    )
    return variants


# ACTB (beta-actin) first 50 amino acids as reference
ACTB_PROTEIN = "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQS"

@app.get("/api/reference/actb")
async def get_actb_reference():
    """Get ACTB reference with synonymous variants."""
    if AppState.loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    variants = AppState.loader.generate_synonymous_variants(ACTB_PROTEIN, n_variants=3)

    return {
        "gene": "ACTB",
        "description": "Beta-actin (first 60 AA)",
        "protein": ACTB_PROTEIN,
        "variants": variants
    }


# =============================================================================
# MAIN
# =============================================================================
def run(host: str = None, port: int = None, model_path: Path = None):
    """Run the visualization server."""
    # Set model path before server starts (lifespan will use it)
    if model_path:
        Config.set_model_path(model_path)

    host = host or VIS_CONFIG.get("host", "127.0.0.1")
    port = port or VIS_CONFIG.get("port", 8765)

    print(f"\n{'='*50}")
    print("Codon Encoder Visualizer")
    print(f"{'='*50}")
    print(f"Model: {Config.get_model_path()}")
    print(f"Server: http://{host}:{port}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
