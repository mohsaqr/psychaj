/**
 * EBIC GLASSO path (matches qgraph::EBICglasso() within 1e-4)
 *
 * Optimized: during the lambda path scan, uses glassoCD (no Theta allocation)
 * and computes EBIC directly from W/Beta. Theta is only constructed once
 * for the best lambda.
 */

import { glassoCD, buildTheta, runGlasso } from './glasso';
import { logDet } from '../core/linalg';

/**
 * Compute EBIC score directly from W and Beta without constructing Theta.
 *
 * Uses the identity: tr(R @ Theta) = sum_j thetaJJ_j * (R[j][j] - sum_{a≠j} R[a][j] * Beta[j][a])
 * where thetaJJ_j = 1 / (W[j][j] - sum_{a≠j} W[a][j] * Beta[j][a]).
 *
 * Edge count uses the symmetrized Theta threshold:
 * |(-Beta[j][i]*thetaJJ_j - Beta[i][j]*thetaJJ_i) / 2| > 1e-8
 */
function computeEbicFromWB(
  R: number[][],
  W: number[][],
  Beta: number[][],
  n: number,
  gamma: number,
): number {
  const p = R.length;

  // log|Theta| = -log|W|
  const ldW = logDet(W);
  if (!isFinite(ldW)) return Infinity;
  const ld = -ldW;

  // Compute thetaJJ for each j and tr(R @ Theta)
  const thetaJJ = new Float64Array(p);
  let trRTheta = 0;

  for (let j = 0; j < p; j++) {
    let dot = 0;
    let rDotBeta = 0;
    for (let a = 0; a < p; a++) {
      if (a !== j) {
        dot += W[a]![j]! * Beta[j]![a]!;
        rDotBeta += R[a]![j]! * Beta[j]![a]!;
      }
    }
    const denom = W[j]![j]! - dot;
    thetaJJ[j] = Math.abs(denom) > 1e-12 ? 1 / denom : 1e6;
    trRTheta += thetaJJ[j]! * (R[j]![j]! - rDotBeta);
  }

  // Count edges (off-diagonal non-zeros in symmetrized Theta)
  let edgeCount = 0;
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const sym = Math.abs(Beta[j]![i]! * thetaJJ[j]! + Beta[i]![j]! * thetaJJ[i]!) / 2;
      if (sym > 1e-8) edgeCount++;
    }
  }

  const loglik = (n / 2) * (ld - trRTheta);
  return -2 * loglik + Math.log(n) * edgeCount + 4 * gamma * Math.log(p) * edgeCount;
}

export function ebicGlasso(
  R: number[][],
  n: number,
  gamma = 0.5,
  nLambda = 100,
  customLambdas?: number[],
  skipRefinement = false,
): { Theta: number[][]; lambda: number } {
  const p = R.length;

  let lambdas: number[];
  if (customLambdas && customLambdas.length > 0) {
    lambdas = customLambdas;
  } else {
    let lambdaMax = 0;
    for (let i = 0; i < p; i++) {
      for (let j = i + 1; j < p; j++) {
        const v = Math.abs(R[i]![j]!);
        if (v > lambdaMax) lambdaMax = v;
      }
    }
    const lambdaMin = 0.01 * lambdaMax;
    if (lambdaMax < 1e-12) {
      const Theta = Array.from({ length: p }, (_, i) =>
        Array.from({ length: p }, (__, j) => (i === j ? 1 : 0)),
      );
      return { Theta, lambda: 0 };
    }
    const logMax = Math.log(lambdaMax);
    const logMin = Math.log(lambdaMin);
    lambdas = Array.from({ length: nLambda }, (_, k) =>
      Math.exp(logMax + (logMin - logMax) * (k / (nLambda - 1))),
    );
  }

  let bestEBIC = Infinity;
  let bestLambda = lambdas[0]!;
  let bestW: number[][] | null = null;
  let bestBeta: number[][] | null = null;

  // Warm-start path: each lambda initialized from the previous solution
  let prevW: number[][] | undefined;
  let prevBeta: number[][] | undefined;

  // Early stopping: once EBIC increases for 10+ consecutive lambdas past
  // the minimum, the U-shaped curve won't come back down. Safe to stop.
  let consecutiveIncreases = 0;

  for (const lam of lambdas) {
    // Use glassoCD with inPlace=true — avoids copying W/Beta each lambda.
    // The path feeds each result as warm-start to the next, so mutation is safe.
    const { W, Beta } = glassoCD(R, lam, 100, 1e-4, 200, 1e-6, prevW, prevBeta, !!prevW);
    prevW = W;
    prevBeta = Beta;

    const ebic = computeEbicFromWB(R, W, Beta, n, gamma);

    if (ebic < bestEBIC) {
      bestEBIC = ebic;
      bestLambda = lam;
      // Must copy since in-place mode mutates W/Beta on next lambda
      bestW = W.map(row => [...row]);
      bestBeta = Beta.map(row => [...row]);
      consecutiveIncreases = 0;
    } else {
      consecutiveIncreases++;
      if (consecutiveIncreases >= 10) break;
    }
  }

  // Construct Theta only for the best lambda
  if (!bestW || !bestBeta) {
    const Theta = Array.from({ length: p }, (_, i) =>
      Array.from({ length: p }, (__, j) => (i === j ? 1 : 0)),
    );
    return { Theta, lambda: bestLambda };
  }

  if (!skipRefinement) {
    // Refinement pass: re-run at selected lambda with tight tolerance
    const { Theta } = runGlasso(
      R, bestLambda, 10000, 1e-8, 500, 1e-10,
      bestW, bestBeta,
    );
    return { Theta, lambda: bestLambda };
  }

  // Build Theta from best W/Beta
  const Theta = buildTheta(bestW, bestBeta);
  return { Theta, lambda: bestLambda };
}
