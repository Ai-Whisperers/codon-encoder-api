/**
 * Codon Inference API - TypeScript Usage Examples
 *
 * These examples demonstrate how to consume the Codon Encoder inference service
 * using the TypeScript client. No model weights or raw embeddings are exposed.
 *
 * Usage:
 *   npx ts-node examples.ts
 */

import { CodonClient, encodeSequence } from './client';
import type {
  EncodedCodon,
  SynonymousVariant,
  DepthStats,
} from './types';

// =============================================================================
// Example 1: Basic Sequence Encoding
// =============================================================================

async function exampleBasicEncoding(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Example 1: Basic Sequence Encoding');
  console.log('='.repeat(60));

  const client = new CodonClient();

  // Encode a short DNA sequence
  const sequence = 'ATGGCTCTGTGG';
  const result = await client.encode(sequence);

  console.log(`Sequence: ${sequence}`);
  console.log(`Codons found: ${result.length}`);
  console.log();

  result.forEach((codon) => {
    console.log(
      `  ${codon.codon} -> ${codon.amino_acid.padEnd(3)} | ` +
      `depth=${codon.depth} | ` +
      `conf=${codon.confidence.toFixed(2)} | ` +
      `margin=${codon.margin.toFixed(3)}`
    );
  });
}

// =============================================================================
// Example 2: Synonymous Variants
// =============================================================================

async function exampleSynonymousVariants(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('Example 2: Synonymous Variants');
  console.log('='.repeat(60));

  const client = new CodonClient();

  // Generate variants for a short peptide
  const protein = 'MDDII';
  const result = await client.getSynonymousVariants(protein, 3);

  console.log(`Protein: ${result.protein}`);
  console.log();

  result.variants.forEach((variant) => {
    console.log(`Strategy: ${variant.strategy}`);
    console.log(`  DNA: ${variant.dna}`);
    console.log(`  Mean depth: ${variant.stats.mean_depth.toFixed(2)}`);
    console.log(`  Depth std: ${variant.stats.depth_std.toFixed(2)}`);
    console.log(`  Mean confidence: ${variant.stats.mean_confidence.toFixed(2)}`);
    console.log();
  });
}

// =============================================================================
// Example 3: Codon Lookup
// =============================================================================

async function exampleLookupCodon(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Example 3: Codon Lookup');
  console.log('='.repeat(60));

  const client = new CodonClient();

  // Look up the start codon
  const codonInfo = await client.getCodon('ATG');

  console.log(`Codon: ${codonInfo.codon}`);
  console.log(`  Amino acid: ${codonInfo.amino_acid}`);
  console.log(`  Position: ${codonInfo.position}`);
  console.log(`  Depth: ${codonInfo.depth}`);
  console.log(`  Poincaré radius: ${codonInfo.poincare_radius.toFixed(3)}`);
  console.log(`  Cluster: ${codonInfo.cluster_idx}`);
  console.log(`  Confidence: ${codonInfo.confidence.toFixed(3)}`);
  console.log(`  Margin: ${codonInfo.margin.toFixed(3)}`);
  console.log(
    `  Projection: (${codonInfo.projection[0].toFixed(3)}, ` +
    `${codonInfo.projection[1].toFixed(3)}, ${codonInfo.projection[2].toFixed(3)})`
  );
}

// =============================================================================
// Example 4: Cluster Analysis
// =============================================================================

async function exampleClusterAnalysis(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('Example 4: Cluster Analysis');
  console.log('='.repeat(60));

  const client = new CodonClient();

  // Get cluster for a specific amino acid
  const cluster = await client.getCluster(0);

  console.log(`Cluster ${cluster.cluster_idx}: ${cluster.amino_acid}`);
  console.log(`  Synonymous codons: ${cluster.codons.length}`);
  console.log();

  cluster.codons.forEach((c) => {
    console.log(`    ${c.codon}: depth=${c.depth}, conf=${c.confidence.toFixed(2)}`);
  });
}

// =============================================================================
// Example 5: Depth Stratification
// =============================================================================

async function exampleDepthStratification(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('Example 5: Depth Stratification');
  console.log('='.repeat(60));

  const client = new CodonClient();

  // Get codons at depth 0 (outermost)
  const depthInfo = await client.getDepth(0);

  console.log(`Depth ${depthInfo.depth}`);
  console.log(`  Poincaré radius: ${depthInfo.poincare_radius.toFixed(3)}`);
  console.log(`  Codons at this depth: ${depthInfo.codons.length}`);
  console.log();

  // Group by amino acid
  const byAA: Record<string, string[]> = {};
  depthInfo.codons.forEach((c) => {
    if (!byAA[c.amino_acid]) byAA[c.amino_acid] = [];
    byAA[c.amino_acid].push(c.codon);
  });

  Object.entries(byAA)
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([aa, codons]) => {
      console.log(`    ${aa}: ${codons.join(', ')}`);
    });
}

// =============================================================================
// Example 6: Angular Variance Analysis
// =============================================================================

async function exampleAngularVariance(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('Example 6: Angular Variance Analysis');
  console.log('='.repeat(60));

  const client = new CodonClient();

  const variance = await client.getAngularVariance();

  console.log('Summary:');
  console.log(`  Mean intra-AA variance: ${variance.summary.mean_intra.toFixed(4)}`);
  console.log(`  Mean inter-AA variance: ${variance.summary.mean_inter.toFixed(4)}`);
  console.log(`  Overall ratio: ${variance.summary.overall_ratio.toFixed(2)}`);
  console.log();

  console.log('By depth:');
  Object.entries(variance.by_depth)
    .sort(([a], [b]) => Number(a) - Number(b))
    .forEach(([depth, stats]) => {
      const aas = (stats as DepthStats).amino_acids.slice(0, 3).join(', ');
      console.log(`  Depth ${depth}: ratio=${(stats as DepthStats).ratio.toFixed(2)}, AAs=[${aas}...]`);
    });
}

// =============================================================================
// Example 7: Visualization Data
// =============================================================================

async function exampleVisualizationData(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('Example 7: Visualization Data');
  console.log('='.repeat(60));

  const client = new CodonClient();

  const viz = await client.getVisualization();

  console.log(`Points: ${viz.points.length}`);
  console.log(`Edges: ${viz.edges.length}`);
  console.log(`Model version: ${viz.metadata.version}`);
  console.log(`Structure score: ${viz.metadata.structure_score.toFixed(4)}`);
  console.log(`Cluster accuracy: ${(viz.metadata.cluster_accuracy * 100).toFixed(1)}%`);
  console.log(`Projection method: ${viz.metadata.projection_method}`);
}

// =============================================================================
// Example 8: Quick Encode Function
// =============================================================================

async function exampleQuickEncode(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('Example 8: Quick Encode Function');
  console.log('='.repeat(60));

  // One-liner encoding
  const codons = await encodeSequence('ATGGAAGAGTGA');

  console.log(`Encoded ${codons.length} codons:`);
  codons.forEach((c) => {
    const aa = c.amino_acid !== '*' ? c.amino_acid : 'STOP';
    console.log(`  ${c.codon} = ${aa}`);
  });
}

// =============================================================================
// Example 9: React/Next.js Integration
// =============================================================================

/**
 * Example React hook for using the Codon API
 *
 * ```tsx
 * import { useState, useCallback } from 'react';
 * import { CodonClient, EncodedCodon } from 'codon-client';
 *
 * export function useCodonEncoder() {
 *   const [result, setResult] = useState<EncodedCodon[]>([]);
 *   const [loading, setLoading] = useState(false);
 *   const [error, setError] = useState<Error | null>(null);
 *
 *   const client = useMemo(() => new CodonClient(), []);
 *
 *   const encode = useCallback(async (sequence: string) => {
 *     setLoading(true);
 *     setError(null);
 *     try {
 *       const data = await client.encode(sequence);
 *       setResult(data);
 *     } catch (e) {
 *       setError(e as Error);
 *     } finally {
 *       setLoading(false);
 *     }
 *   }, [client]);
 *
 *   return { result, loading, error, encode };
 * }
 * ```
 */

// =============================================================================
// Example 10: Local Development
// =============================================================================

async function exampleLocalDevelopment(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log('Example 10: Local Development');
  console.log('='.repeat(60));

  // Connect to local server
  const client = new CodonClient({ baseUrl: 'http://localhost:8000' });

  console.log('Connecting to local server at http://localhost:8000');
  console.log('(Skipping actual request - run visualizer locally to test)');
}

// =============================================================================
// Run all examples
// =============================================================================

async function main(): Promise<void> {
  console.log();
  console.log('Codon Inference API - TypeScript Client Examples');
  console.log('Note: These examples require the API server to be running.');
  console.log('      For local testing, start the visualizer server first.');
  console.log();

  // Uncomment to run examples against a running server:
  // await exampleBasicEncoding();
  // await exampleSynonymousVariants();
  // await exampleLookupCodon();
  // await exampleClusterAnalysis();
  // await exampleDepthStratification();
  // await exampleAngularVariance();
  // await exampleVisualizationData();
  // await exampleQuickEncode();
  // await exampleLocalDevelopment();

  console.log('Examples are defined but not executed.');
  console.log('Uncomment the function calls above to run against a live server.');
}

main().catch(console.error);
