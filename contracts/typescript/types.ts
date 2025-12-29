/**
 * Codon Inference API - TypeScript Client Types
 *
 * API endpoint: https://codon.ai-whisperers.org/api
 *
 * These types define the public contract for consuming the Codon Encoder
 * inference service. Model weights and raw embeddings are not exposed.
 */

// =============================================================================
// Core Data Types
// =============================================================================

/**
 * Single codon with inference results
 *
 * Note: Raw 16-dim embeddings are NOT exposed. Only derived metrics
 * (projection, confidence, margin) are available.
 */
export interface CodonPoint {
  /** Three-letter codon (e.g., "ATG") */
  codon: string;

  /** Single-letter amino acid code (e.g., "M" for Methionine) */
  amino_acid: string;

  /** Position index in the canonical 64-codon ordering (0-63) */
  position: number;

  /** Hierarchical depth level (0-9) */
  depth: number;

  /** 3D projection coordinates for visualization */
  projection: [number, number, number];

  /** Euclidean norm of the internal embedding (scalar, not the vector) */
  embedding_norm: number;

  /** Hyperbolic radius in Poincaré disk (0.1 to 0.9, inversely correlated with depth) */
  poincare_radius: number;

  /** Predicted amino acid cluster index (0-20) */
  cluster_idx: number;

  /** Cluster assignment confidence: 1/(1+distance_to_nearest_cluster) */
  confidence: number;

  /** Decision margin: distance_to_second_nearest - distance_to_nearest */
  margin: number;

  /** Display color (hex string, e.g., "#FF5733") */
  color: string;
}

/**
 * Edge/fiber connecting two codons in visualization
 */
export interface Edge {
  /** Index of source codon (0-63) */
  source: number;

  /** Index of target codon (0-63) */
  target: number;

  /** Connection strength (0.0 to 1.0) */
  weight: number;
}

/**
 * Model metadata (public information only)
 */
export interface ModelMetadata {
  /** Model version identifier */
  version: string;

  /** Structure quality score (Spearman correlation) */
  structure_score: number;

  /** Cluster classification accuracy */
  cluster_accuracy: number;

  /** Embedding dimensionality (informational, embeddings not exposed) */
  embed_dim: number;

  /** Number of amino acid clusters */
  num_clusters: number;

  /** Projection method used (PCA, UMAP, t-SNE) */
  projection_method: string;
}

// =============================================================================
// API Request Types
// =============================================================================

/**
 * Request body for sequence encoding
 */
export interface EncodeRequest {
  /** DNA sequence (ATCG characters, non-ATCG filtered) */
  sequence: string;
}

/**
 * Request body for synonymous variant generation
 */
export interface SynonymousVariantsRequest {
  /** Protein sequence (single-letter amino acid codes) */
  protein: string;

  /** Number of variants to generate (default: 3) */
  n_variants?: number;
}

// =============================================================================
// API Response Types
// =============================================================================

/**
 * Single codon in an encoded sequence trajectory
 */
export interface EncodedCodon {
  /** Position in the input sequence (0-indexed) */
  seq_position: number;

  /** Three-letter codon */
  codon: string;

  /** Translated amino acid */
  amino_acid: string;

  /** Reference index in canonical 64-codon list */
  ref_idx: number;

  /** 3D projection coordinates */
  projection: [number, number, number];

  /** Hierarchical depth (0-9) */
  depth: number;

  /** Hyperbolic radius */
  poincare_radius: number;

  /** Embedding norm (scalar) */
  embedding_norm: number;

  /** Cluster assignment index */
  cluster_idx: number;

  /** Cluster confidence */
  confidence: number;

  /** Decision margin */
  margin: number;
}

/**
 * Response from POST /api/encode
 */
export type EncodeResponse = EncodedCodon[];

/**
 * Full visualization payload
 */
export interface VisualizationResponse {
  /** All 64 codon points with inference results */
  points: CodonPoint[];

  /** Edge connections (mode-dependent) */
  edges: Edge[];

  /** Visualization configuration */
  config: VisualizationConfig;

  /** Model metadata */
  metadata: ModelMetadata;
}

/**
 * Visualization configuration
 */
export interface VisualizationConfig {
  /** Depth-to-color mapping (10 hex colors) */
  depth_colors: string[];

  /** Amino acid to color mapping */
  amino_acid_colors: Record<string, string>;

  /** Fiber threshold for edge computation */
  fiber_threshold: number;
}

/**
 * Single depth level statistics
 */
export interface DepthStats {
  /** Variance within amino acid groups (radians²) */
  intra_aa_variance: number;

  /** Variance between amino acid centroids (radians²) */
  inter_aa_variance: number;

  /** Ratio: inter/intra (higher = better separation) */
  ratio: number;

  /** Amino acids present at this depth */
  amino_acids: string[];
}

/**
 * Response from GET /api/angular_variance
 */
export interface AngularVarianceResponse {
  /** Statistics per depth level (0-9) */
  by_depth: Record<number, DepthStats>;

  /** Summary statistics */
  summary: {
    mean_intra: number;
    mean_inter: number;
    overall_ratio: number;
  };
}

/**
 * Single synonymous variant
 */
export interface SynonymousVariant {
  /** Strategy used: "random" | "high_depth" | "low_depth" */
  strategy: string;

  /** Generated DNA sequence */
  dna: string;

  /** Encoded trajectory */
  encoded: EncodedCodon[];

  /** Trajectory statistics */
  stats: {
    mean_depth: number;
    depth_std: number;
    mean_confidence: number;
  };
}

/**
 * Response from POST /api/synonymous_variants
 */
export interface SynonymousVariantsResponse {
  /** Original protein sequence */
  protein: string;

  /** Generated variants */
  variants: SynonymousVariant[];
}

/**
 * Response from GET /api/codon/{codon}
 */
export interface SingleCodonResponse {
  codon: string;
  amino_acid: string;
  position: number;
  depth: number;
  projection: [number, number, number];
  poincare_radius: number;
  cluster_idx: number;
  confidence: number;
  margin: number;
}

/**
 * Response from GET /api/cluster/{idx}
 */
export interface ClusterResponse {
  cluster_idx: number;
  amino_acid: string;
  codons: SingleCodonResponse[];
}

/**
 * Response from GET /api/depth/{level}
 */
export interface DepthLevelResponse {
  depth: number;
  poincare_radius: number;
  codons: SingleCodonResponse[];
}

// =============================================================================
// Error Types
// =============================================================================

export interface APIError {
  detail: string;
  status_code: number;
}

// =============================================================================
// Client Configuration
// =============================================================================

export interface ClientConfig {
  /** Base URL for the API (default: https://codon.ai-whisperers.org) */
  baseUrl?: string;

  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;

  /** Optional API key for rate limiting */
  apiKey?: string;
}
