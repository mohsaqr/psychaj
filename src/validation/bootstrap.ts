/**
 * Bootnet-style bootstrap for GLASSO / psychometric networks.
 *
 * Implements:
 *   – Case-resampling bootstrap for edge weight accuracy (CIs)
 *   – Pairwise edge difference testing (CI of A − B excludes 0)
 *
 * Reference: Epskamp, Borsboom & Fried (2018), Psychological Methods.
 */

import { computePearsonMatrix } from '../core/pearson';
import { computePartialCorrMatrix } from '../core/partial-corr';
import { ebicGlasso } from '../estimation/ebic-glasso';
import { thetaToPcor } from '../estimation/theta-to-pcor';
import { runGlasso } from '../estimation/glasso';
import { percentile } from '../core/stats';
import { SeededRNG } from '../core/rng';
import type {
  BootstrapGlassoOptions,
  BootstrapGlassoResult,
  GlassoBootEdge,
  GlassoBootDiff,
} from '../core/types';

function estimatePcor(
  data: number[][],
  method: 'cor' | 'pcor' | 'glasso',
  gamma: number,
  rho: number,
  skipRefinement: boolean,
): { pcor: number[][]; lambda?: number } {
  const R = computePearsonMatrix(data);
  const n = data.length;
  if (method === 'cor') return { pcor: R };
  if (method === 'pcor') return { pcor: computePartialCorrMatrix(R) };
  if (rho > 0) {
    const { Theta } = runGlasso(R, rho);
    return { pcor: thetaToPcor(Theta), lambda: rho };
  }
  const { Theta, lambda } = ebicGlasso(R, n, gamma, 100, undefined, skipRefinement);
  return { pcor: thetaToPcor(Theta), lambda };
}

export function bootstrapGlasso(
  rawData: number[][],
  labels: string[],
  opts: BootstrapGlassoOptions = {},
): BootstrapGlassoResult {
  const method = opts.method ?? 'glasso';
  const gamma = opts.gamma ?? 0.5;
  const rho = opts.rho ?? 0;
  const iter = opts.iter ?? 1000;
  const ciLevel = opts.ciLevel ?? 0.95;
  const alpha = 1 - ciLevel;
  const n = rawData.length;
  const p = labels.length;

  if (n < 4 || p < 2) {
    return { edges: [], pairwiseDiffs: [], labels, n, iter, method, gamma };
  }

  const rng = new SeededRNG(opts.seed ?? 42);

  const { pcor: origPcor, lambda: origLambda } = estimatePcor(
    rawData, method, gamma, rho, false,
  );

  const edgeCount = (p * (p - 1)) / 2;
  const edgeIs: number[] = [];
  const edgeJs: number[] = [];
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      edgeIs.push(i);
      edgeJs.push(j);
    }
  }

  const bootStorage: Float64Array[] = Array.from({ length: edgeCount }, () => new Float64Array(iter));
  const sample: number[][] = Array.from({ length: n }, () => new Array<number>(p).fill(0));

  // For GLASSO with auto lambda (rho=0): fix the lambda selected on the
  // original data for all bootstrap iterations. This avoids re-running the
  // 100-lambda EBIC path search each iteration (~100x speedup).
  // Standard approach matching R bootnet.
  const bootRho = (method === 'glasso' && rho === 0 && origLambda) ? origLambda : rho;

  for (let b = 0; b < iter; b++) {
    for (let i = 0; i < n; i++) {
      const src = rng.randInt(n);
      const srcRow = rawData[src]!;
      const dstRow = sample[i]!;
      for (let jj = 0; jj < p; jj++) dstRow[jj] = srcRow[jj]!;
    }

    const { pcor: bootPcor } = estimatePcor(sample, method, gamma, bootRho, true);

    for (let k = 0; k < edgeCount; k++) {
      bootStorage[k]![b] = bootPcor[edgeIs[k]!]![edgeJs[k]!]!;
    }
  }

  const loP = alpha / 2;
  const hiP = 1 - alpha / 2;

  const edges: GlassoBootEdge[] = edgeIs.map((ei, k) => {
    const ej = edgeJs[k]!;
    const samples = bootStorage[k]!;
    const sorted = samples.slice().sort() as Float64Array;
    const bootMean = samples.reduce((s, v) => s + v, 0) / iter;
    const variance = samples.reduce((s, v) => s + (v - bootMean) ** 2, 0) / (iter - 1);
    const bootSd = Math.sqrt(variance);
    const ciLower = percentile(sorted, loP);
    const ciUpper = percentile(sorted, hiP);
    return {
      from: labels[ei]!,
      to: labels[ej]!,
      i: ei,
      j: ej,
      weight: origPcor[ei]![ej]!,
      bootMean,
      bootSd,
      ciLower,
      ciUpper,
      significant: ciLower > 0 || ciUpper < 0,
      bootSamples: samples,
    };
  });

  edges.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

  const pairwiseDiffs: GlassoBootDiff[] = [];
  const diffSamples = new Float64Array(iter);

  for (let a = 0; a < edges.length; a++) {
    for (let b2 = a + 1; b2 < edges.length; b2++) {
      const edgeA = edges[a]!;
      const edgeB = edges[b2]!;
      for (let t = 0; t < iter; t++) {
        diffSamples[t] = edgeA.bootSamples[t]! - edgeB.bootSamples[t]!;
      }
      const diffSorted = diffSamples.slice().sort() as Float64Array;
      const diffMean = diffSamples.reduce((s, v) => s + v, 0) / iter;
      const diffCiLower = percentile(diffSorted, loP);
      const diffCiUpper = percentile(diffSorted, hiP);
      pairwiseDiffs.push({
        edgeA: `${edgeA.from}||${edgeA.to}`,
        edgeB: `${edgeB.from}||${edgeB.to}`,
        diffMean,
        diffCiLower,
        diffCiUpper,
        significant: diffCiLower > 0 || diffCiUpper < 0,
      });
    }
  }

  return { edges, pairwiseDiffs, labels, n, iter, method, gamma, lambda: origLambda };
}

/** Canonical edge key: "from||to" */
export function edgeKey(from: string, to: string): string {
  return `${from}||${to}`;
}

/** Parse edge key back to [from, to] */
export function parseEdgeKey(key: string): [string, string] {
  const idx = key.indexOf('||');
  if (idx < 0) return [key, ''];
  return [key.slice(0, idx), key.slice(idx + 2)];
}
