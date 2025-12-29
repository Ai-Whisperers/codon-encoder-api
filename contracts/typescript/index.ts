/**
 * Codon Inference API - TypeScript Client
 *
 * Open-source client for consuming the Codon Encoder inference service.
 * Model weights and raw embeddings are not exposed through this API.
 *
 * API endpoint: https://codon.ai-whisperers.org/api
 *
 * @example
 * ```typescript
 * import { CodonClient, encodeSequence } from 'codon-client';
 *
 * // Quick encoding
 * const codons = await encodeSequence("ATGGCTCTGTGG");
 *
 * // Full client
 * const client = new CodonClient();
 * const result = await client.encode("ATGGCTCTGTGG");
 * const metadata = await client.getMetadata();
 * ```
 */

export { CodonClient, CodonClientError, encodeSequence } from './client';

export type {
  // Core types
  CodonPoint,
  Edge,
  ModelMetadata,
  ClientConfig,
  APIError,

  // Request types
  EncodeRequest,
  SynonymousVariantsRequest,

  // Response types
  EncodedCodon,
  EncodeResponse,
  VisualizationResponse,
  VisualizationConfig,
  AngularVarianceResponse,
  DepthStats,
  SynonymousVariant,
  SynonymousVariantsResponse,
  SingleCodonResponse,
  ClusterResponse,
  DepthLevelResponse,
} from './types';
