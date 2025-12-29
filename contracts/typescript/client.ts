/**
 * Codon Inference API - TypeScript Client
 *
 * API endpoint: https://codon.ai-whisperers.org/api
 *
 * This client provides a type-safe interface for consuming the Codon Encoder
 * inference service. Model weights and raw embeddings are not exposed.
 *
 * @example
 * ```typescript
 * import { CodonClient } from './client';
 *
 * const client = new CodonClient();
 * const result = await client.encode("ATGGCTCTGTGG");
 * result.forEach(codon => {
 *   console.log(`${codon.codon} -> ${codon.amino_acid} (depth=${codon.depth})`);
 * });
 * ```
 */

import type {
  AngularVarianceResponse,
  ClientConfig,
  ClusterResponse,
  CodonPoint,
  DepthLevelResponse,
  Edge,
  EncodedCodon,
  ModelMetadata,
  SingleCodonResponse,
  SynonymousVariantsResponse,
  VisualizationResponse,
} from './types';

export class CodonClientError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: unknown
  ) {
    super(message);
    this.name = 'CodonClientError';
  }
}

export class CodonClient {
  private baseUrl: string;
  private timeout: number;
  private apiKey?: string;

  static readonly DEFAULT_BASE_URL = 'https://codon.ai-whisperers.org';

  constructor(config: ClientConfig = {}) {
    this.baseUrl = (config.baseUrl ?? CodonClient.DEFAULT_BASE_URL).replace(/\/$/, '');
    this.timeout = config.timeout ?? 30000;
    this.apiKey = config.apiKey;
  }

  private async request<T>(
    method: string,
    endpoint: string,
    options: { json?: unknown; params?: Record<string, string> } = {}
  ): Promise<T> {
    const url = new URL(`/api${endpoint}`, this.baseUrl);

    if (options.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        url.searchParams.append(key, value);
      });
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url.toString(), {
        method,
        headers,
        body: options.json ? JSON.stringify(options.json) : undefined,
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new CodonClientError(
          `API error: ${text}`,
          response.status,
          text
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof CodonClientError) throw error;
      if (error instanceof Error && error.name === 'AbortError') {
        throw new CodonClientError('Request timeout');
      }
      throw new CodonClientError(`Request failed: ${error}`);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ===========================================================================
  // Core Endpoints
  // ===========================================================================

  /**
   * Get complete visualization data.
   *
   * Returns all 64 codon points, edges, configuration, and metadata.
   */
  async getVisualization(): Promise<VisualizationResponse> {
    return this.request<VisualizationResponse>('GET', '/visualization');
  }

  /**
   * Get all 64 codon points with inference results.
   */
  async getPoints(): Promise<CodonPoint[]> {
    return this.request<CodonPoint[]>('GET', '/points');
  }

  /**
   * Get fiber/edge connections between codons.
   *
   * @param mode - Connection mode:
   *   - hierarchical: same depth, embedding distance < threshold
   *   - amino_acid: synonymous codons (same translation)
   *   - depth: adjacent depth levels
   *   - none: no edges
   */
  async getEdges(
    mode: 'hierarchical' | 'amino_acid' | 'depth' | 'none' = 'hierarchical'
  ): Promise<Edge[]> {
    return this.request<Edge[]>('GET', '/edges', { params: { mode } });
  }

  /**
   * Get model metadata.
   */
  async getMetadata(): Promise<ModelMetadata> {
    return this.request<ModelMetadata>('GET', '/metadata');
  }

  // ===========================================================================
  // Inference Endpoints
  // ===========================================================================

  /**
   * Encode a DNA sequence into codon trajectory.
   *
   * Takes a DNA sequence and returns inference results for each codon,
   * including 3D projections, cluster assignments, and confidence scores.
   *
   * @param sequence - DNA sequence (ATCG characters, non-ATCG filtered)
   *
   * @example
   * ```typescript
   * const result = await client.encode("ATGGCTCTGTGG");
   * result.forEach(codon => {
   *   console.log(`${codon.codon}: depth=${codon.depth}, conf=${codon.confidence.toFixed(2)}`);
   * });
   * ```
   */
  async encode(sequence: string): Promise<EncodedCodon[]> {
    return this.request<EncodedCodon[]>('POST', '/encode', {
      json: { sequence },
    });
  }

  /**
   * Generate synonymous DNA variants for a protein sequence.
   *
   * Creates alternative DNA sequences that encode the same protein
   * using different codon choices, with various selection strategies.
   *
   * @param protein - Protein sequence (single-letter amino acid codes)
   * @param nVariants - Number of variants to generate (default: 3)
   *
   * @example
   * ```typescript
   * const result = await client.getSynonymousVariants("MDDII", 3);
   * result.variants.forEach(v => {
   *   console.log(`${v.strategy}: ${v.dna} (mean_depth=${v.stats.mean_depth.toFixed(2)})`);
   * });
   * ```
   */
  async getSynonymousVariants(
    protein: string,
    nVariants: number = 3
  ): Promise<SynonymousVariantsResponse> {
    return this.request<SynonymousVariantsResponse>('POST', '/synonymous_variants', {
      json: { protein, n_variants: nVariants },
    });
  }

  // ===========================================================================
  // Lookup Endpoints
  // ===========================================================================

  /**
   * Get inference results for a single codon.
   *
   * @param codon - Three-letter codon (e.g., "ATG")
   */
  async getCodon(codon: string): Promise<SingleCodonResponse> {
    return this.request<SingleCodonResponse>('GET', `/codon/${codon.toUpperCase()}`);
  }

  /**
   * Get all codons in an amino acid cluster.
   *
   * @param idx - Cluster index (0-20)
   */
  async getCluster(idx: number): Promise<ClusterResponse> {
    return this.request<ClusterResponse>('GET', `/cluster/${idx}`);
  }

  /**
   * Get all codons at a specific depth level.
   *
   * @param level - Depth level (0-9)
   */
  async getDepth(level: number): Promise<DepthLevelResponse> {
    return this.request<DepthLevelResponse>('GET', `/depth/${level}`);
  }

  // ===========================================================================
  // Analytics Endpoints
  // ===========================================================================

  /**
   * Get angular variance statistics by depth level.
   *
   * Returns circular variance metrics measuring the separation
   * quality between and within amino acid groups at each depth.
   */
  async getAngularVariance(): Promise<AngularVarianceResponse> {
    return this.request<AngularVarianceResponse>('GET', '/angular_variance');
  }

  /**
   * Get reference ACTB gene with synonymous variants.
   *
   * Returns the first 60 amino acids of beta-actin (ACTB)
   * with three encoded variants for comparison.
   */
  async getReferenceActb(): Promise<SynonymousVariantsResponse> {
    return this.request<SynonymousVariantsResponse>('GET', '/reference/actb');
  }
}

// Convenience function for quick encoding
export async function encodeSequence(
  sequence: string,
  baseUrl?: string
): Promise<EncodedCodon[]> {
  const client = new CodonClient({ baseUrl });
  return client.encode(sequence);
}

export default CodonClient;
