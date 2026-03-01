/**
 * Centrality stability estimation via case-dropping bootstrap.
 * Uses a callback pattern: decoupled from any specific estimation method.
 *
 * Usage:
 *   estimateCS(freqMatrix, (subset) => {
 *     const R = computePearsonMatrix(subset);
 *     const { Theta } = ebicGlasso(R, subset.length);
 *     return thetaToPcor(Theta);
 *   });
 */

import { SeededRNG } from '../core/rng';
import { pearsonCorr } from '../core/stats';
import type { StabilityResult, StabilityOptions } from '../core/types';
import { computeCentralities } from '../graph/centrality';

/** Network estimation function: given a data subset, return a weight matrix. */
export type NetworkFn = (data: number[][]) => number[][];

/**
 * Estimate centrality stability using case-dropping bootstrap.
 *
 * @param data        n × p data matrix (rows = observations, cols = variables)
 * @param networkFn   callback that estimates a network from a data subset
 * @param options     bootstrap options
 */
export function estimateCS(
  data: number[][],
  networkFn: NetworkFn,
  options: StabilityOptions = {},
): StabilityResult {
  const {
    measures = ['inStrength', 'outStrength', 'betweenness'],
    iter = 500,
    dropProps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    threshold = 0.7,
    certainty = 0.95,
    seed = 42,
    corrMethod = 'pearson',
  } = options;

  const n = data.length;
  const p = data[0]?.length ?? 0;
  if (n < 4 || p < 2) {
    return {
      csCoefficients: Object.fromEntries(measures.map(m => [m, 0])),
      meanCorrelations: Object.fromEntries(measures.map(m => [m, dropProps.map(() => NaN)])),
      dropProps,
      threshold,
      certainty,
    };
  }

  const rng = new SeededRNG(seed);

  // Compute original network and centralities
  let origWeights: number[][];
  try {
    origWeights = networkFn(data);
  } catch {
    return {
      csCoefficients: Object.fromEntries(measures.map(m => [m, 0])),
      meanCorrelations: Object.fromEntries(measures.map(m => [m, dropProps.map(() => NaN)])),
      dropProps,
      threshold,
      certainty,
    };
  }

  const origCent = computeCentralities(origWeights);

  // Map measure names to centrality arrays
  const getMeasureValues = (cent: ReturnType<typeof computeCentralities>, measure: string): Float64Array | null => {
    switch (measure) {
      case 'inStrength': return cent.inStrength;
      case 'outStrength': return cent.outStrength;
      case 'betweenness': return cent.betweenness;
      case 'closeness': return cent.closeness;
      default: return null;
    }
  };

  // Check which measures have non-zero variance
  const validMeasures: string[] = [];
  for (const m of measures) {
    const vals = getMeasureValues(origCent, m);
    if (!vals) continue;
    let mean = 0;
    for (let i = 0; i < p; i++) mean += vals[i]!;
    mean /= p;
    let variance = 0;
    for (let i = 0; i < p; i++) variance += (vals[i]! - mean) ** 2;
    if (variance > 0) validMeasures.push(m);
  }

  const correlations: Record<string, number[][]> = {};
  for (const m of validMeasures) {
    correlations[m] = dropProps.map(() => []);
  }

  // Case-dropping bootstrap
  for (let j = 0; j < dropProps.length; j++) {
    const dp = dropProps[j]!;
    const nDrop = Math.floor(n * dp);
    const nKeep = n - nDrop;
    if (nDrop === 0 || nKeep < 2) continue;

    for (let it = 0; it < iter; it++) {
      const keepIdx = rng.choiceWithoutReplacement(n, nKeep);
      const subset = keepIdx.map(i => data[i]!);

      let subWeights: number[][];
      try {
        subWeights = networkFn(subset);
      } catch {
        continue;
      }

      const subCent = computeCentralities(subWeights);

      for (const m of validMeasures) {
        const origVals = getMeasureValues(origCent, m)!;
        const subVals = getMeasureValues(subCent, m)!;
        const corr = corrMethod === 'spearman'
          ? spearmanCorr(origVals, subVals)
          : pearsonCorr(origVals, subVals);
        correlations[m]![j]!.push(corr);
      }
    }
  }

  // Compute mean correlations and CS coefficients
  const meanCorrelations: Record<string, number[]> = {};
  const csCoefficients: Record<string, number> = {};

  for (const m of measures) {
    if (validMeasures.includes(m)) {
      const means: number[] = [];
      for (let j = 0; j < dropProps.length; j++) {
        const corrs = correlations[m]![j]!;
        const valid = corrs.filter(c => !isNaN(c));
        if (valid.length === 0) {
          means.push(NaN);
          continue;
        }
        means.push(valid.reduce((s, v) => s + v, 0) / valid.length);
      }
      meanCorrelations[m] = means;

      let cs = 0;
      for (let j = 0; j < dropProps.length; j++) {
        const corrs = correlations[m]![j]!;
        const validCorrs = corrs.filter(c => !isNaN(c));
        if (validCorrs.length === 0) continue;
        const aboveThreshold = validCorrs.filter(c => c >= threshold).length / validCorrs.length;
        if (aboveThreshold >= certainty) {
          cs = dropProps[j]!;
        }
      }
      csCoefficients[m] = cs;
    } else {
      meanCorrelations[m] = dropProps.map(() => NaN);
      csCoefficients[m] = 0;
    }
  }

  return {
    csCoefficients,
    meanCorrelations,
    dropProps,
    threshold,
    certainty,
  };
}

/** Rank array values (average ranks for ties). */
function rankArray(arr: Float64Array): Float64Array {
  const indexed = Array.from(arr, (v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Float64Array(arr.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j]!.v === indexed[i]!.v) j++;
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) ranks[indexed[k]!.i] = avgRank;
    i = j;
  }
  return ranks;
}

/** Spearman rank correlation = Pearson correlation on ranks. */
function spearmanCorr(a: Float64Array, b: Float64Array): number {
  return pearsonCorr(rankArray(a), rankArray(b));
}
