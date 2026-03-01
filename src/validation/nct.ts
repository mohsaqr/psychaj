/**
 * Network Comparison Test (NCT)
 * Permutation-based comparison of two psychometric networks.
 *
 * Reference: van Borkulo et al. (2022). Comparing network structures on three
 * aspects: A permutation test. Psychological Methods.
 */

import { SeededRNG } from '../core/rng';
import { computePearsonMatrix } from '../core/pearson';
import { computePartialCorrMatrix } from '../core/partial-corr';
import { ebicGlasso } from '../estimation/ebic-glasso';
import { thetaToPcor } from '../estimation/theta-to-pcor';
import { pAdjust } from './p-adjust';
import type { NctOptions, NctEdgeResult, NctResult } from '../core/types';

export function computeNctLambdaPath(pooledData: number[][], nLambda = 50): number[] {
  const R = computePearsonMatrix(pooledData);
  const p = R.length;
  let lambdaMax = 0;
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const v = Math.abs(R[i]![j]!);
      if (v > lambdaMax) lambdaMax = v;
    }
  }
  if (lambdaMax < 1e-12) return [];
  const lambdaMin = lambdaMax * 0.01;
  const logMax = Math.log(lambdaMax);
  const logMin = Math.log(lambdaMin);
  return Array.from({ length: nLambda }, (_, k) =>
    Math.exp(logMax + (logMin - logMax) * (k / (nLambda - 1))),
  );
}

function estimateNet(
  data: number[][],
  method: 'cor' | 'pcor' | 'glasso',
  gamma: number,
  permLambdas?: number[],
): number[][] | null {
  try {
    const n = data.length;
    if (n < 3) return null;
    const p = data[0]?.length ?? 0;
    if (p < 2) return null;

    const R = computePearsonMatrix(data);

    if (method === 'cor') {
      const net = R.map(row => [...row]);
      for (let i = 0; i < p; i++) net[i]![i] = 0;
      return net;
    }

    if (method === 'pcor') {
      return computePartialCorrMatrix(R);
    }

    const { Theta } = permLambdas && permLambdas.length > 0
      ? ebicGlasso(R, n, gamma, undefined, permLambdas, true)
      : ebicGlasso(R, n, gamma);

    return thetaToPcor(Theta);
  } catch {
    return null;
  }
}

export function networkComparisonTest(
  xData: number[][],
  yData: number[][],
  options: NctOptions = {},
): NctResult {
  const {
    method = 'glasso',
    iter = 1000,
    alpha = 0.05,
    gamma = 0.5,
    paired = false,
    adjust = 'none',
    seed = 42,
    nodeNames,
    permutationIndices,
  } = options;

  const nX = xData.length;
  const nY = yData.length;
  const p = xData[0]?.length ?? 0;

  if (nX < 3) throw new Error(`NCT: group X must have at least 3 observations (got ${nX})`);
  if (nY < 3) throw new Error(`NCT: group Y must have at least 3 observations (got ${nY})`);
  if (p < 2) throw new Error(`NCT: at least 2 variables required (got ${p})`);
  const pY = yData[0]?.length ?? 0;
  if (pY !== p) throw new Error(`NCT: group X has ${p} columns but group Y has ${pY}`);
  if (paired && nX !== nY) {
    throw new Error('NCT: paired test requires equal group sizes');
  }

  const names = nodeNames ?? Array.from({ length: p }, (_, i) => `V${i + 1}`);
  if (names.length !== p) throw new Error('NCT: nodeNames length must match number of columns');

  const rng = new SeededRNG(seed);

  const netX = estimateNet(xData, method, gamma);
  const netY = estimateNet(yData, method, gamma);
  if (!netX || !netY) {
    throw new Error('NCT: failed to estimate networks from input data');
  }

  const obsDiff: number[][] = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (__, j) => netX[i]![j]! - netY[i]![j]!),
  );

  const obsFlat = new Float64Array(p * p);
  const absObsFlat = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      const k = i * p + j;
      obsFlat[k] = obsDiff[i]![j]!;
      absObsFlat[k] = Math.abs(obsDiff[i]![j]!);
    }
  }

  const pooled = [...xData, ...yData];
  const nTotal = nX + nY;

  let permLambdas: number[] | undefined;
  if (method === 'glasso') {
    permLambdas = computeNctLambdaPath(pooled, 50);
  }

  const exceedCounts = new Int32Array(p * p);
  const sumDiffs = new Float64Array(p * p);
  const sumDiffsSq = new Float64Array(p * p);
  let validIter = 0;

  for (let it = 0; it < iter; it++) {
    let idxX: number[];
    let idxY: number[];

    if (permutationIndices && it < permutationIndices.length) {
      idxX = permutationIndices[it]!;
      const idxSet = new Set(idxX);
      idxY = Array.from({ length: nTotal }, (_, i) => i).filter(i => !idxSet.has(i));
    } else if (paired) {
      idxX = Array.from({ length: nX }, (_, i) => i);
      idxY = Array.from({ length: nY }, (_, i) => nX + i);
      for (let pair = 0; pair < nX; pair++) {
        if (rng.random() < 0.5) {
          [idxX[pair], idxY[pair]] = [idxY[pair]!, idxX[pair]!];
        }
      }
    } else {
      const perm = rng.permutation(nTotal);
      idxX = perm.slice(0, nX);
      idxY = perm.slice(nX);
    }

    const permXData = idxX.map(i => pooled[i]!);
    const permYData = idxY.map(i => pooled[i]!);

    const permNetX = estimateNet(permXData, method, gamma, permLambdas);
    const permNetY = estimateNet(permYData, method, gamma, permLambdas);

    if (!permNetX || !permNetY) continue;

    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        const k = i * p + j;
        const diff = permNetX[i]![j]! - permNetY[i]![j]!;
        sumDiffs[k] += diff;
        sumDiffsSq[k] += diff * diff;
        if (Math.abs(diff) >= absObsFlat[k]!) {
          exceedCounts[k]++;
        }
      }
    }
    validIter++;
  }

  const rawPFlat = Array.from({ length: p * p }, (_, k) =>
    (exceedCounts[k]! + 1) / (iter + 1),
  );
  const adjPFlat = pAdjust(rawPFlat, adjust);

  const pValMatrix: number[][] = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (__, j) => adjPFlat[i * p + j]!),
  );

  const n = validIter > 0 ? validIter : 1;
  const effectFlat = new Float64Array(p * p);
  for (let k = 0; k < p * p; k++) {
    const mean = sumDiffs[k]! / n;
    const variance = sumDiffsSq[k]! / n - mean * mean;
    const sd = Math.sqrt(Math.max(variance, 0));
    effectFlat[k] = sd > 0 ? obsFlat[k]! / sd : 0;
  }

  const effectMatrix: number[][] = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (__, j) => effectFlat[i * p + j]!),
  );

  const sigDiff: number[][] = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (__, j) =>
      adjPFlat[i * p + j]! < alpha ? obsDiff[i]![j]! : 0,
    ),
  );

  const summary: NctEdgeResult[] = [];
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const wx = netX[i]![j]!;
      const wy = netY[i]![j]!;
      if (wx === 0 && wy === 0) continue;
      const k = i * p + j;
      summary.push({
        from: names[i]!,
        to: names[j]!,
        weight_x: wx,
        weight_y: wy,
        diff: obsDiff[i]![j]!,
        effect_size: effectFlat[k]!,
        p_value: adjPFlat[k]!,
        significant: adjPFlat[k]! < alpha,
      });
    }
  }

  return {
    obs_diff: obsDiff,
    p_values: pValMatrix,
    effect_size: effectMatrix,
    sig_diff: sigDiff,
    net_x: netX,
    net_y: netY,
    summary,
    nodeNames: names,
    method,
    iter,
    alpha,
    nX,
    nY,
  };
}
