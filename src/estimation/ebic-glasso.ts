/**
 * EBIC GLASSO path (matches qgraph::EBICglasso() within 1e-4)
 */

import { runGlasso } from './glasso';
import { logDet } from '../core/linalg';

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
  let bestTheta: number[][] | null = null;
  let bestLambda = lambdas[0]!;
  let bestW: number[][] | null = null;
  let bestBeta: number[][] | null = null;

  let prevW: number[][] | undefined;
  let prevBeta: number[][] | undefined;

  for (const lam of lambdas) {
    const { Theta, W, Beta } = runGlasso(R, lam, 100, 1e-4, 200, 1e-6, prevW, prevBeta);
    prevW = W;
    prevBeta = Beta;

    let edgeCount = 0;
    for (let i = 0; i < p; i++) {
      for (let j = i + 1; j < p; j++) {
        if (Math.abs(Theta[i]![j]!) > 1e-8) edgeCount++;
      }
    }

    const ldW = logDet(W);
    if (!isFinite(ldW)) continue;
    const ld = -ldW;

    let trRTheta = 0;
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        trRTheta += R[i]![j]! * Theta[i]![j]!;
      }
    }
    const loglik = (n / 2) * (ld - trRTheta);

    const ebic = -2 * loglik + Math.log(n) * edgeCount + 4 * gamma * Math.log(p) * edgeCount;

    if (ebic < bestEBIC) {
      bestEBIC = ebic;
      bestTheta = Theta;
      bestLambda = lam;
      bestW = W;
      bestBeta = Beta;
    }
  }

  if (!bestTheta) {
    bestTheta = Array.from({ length: p }, (_, i) =>
      Array.from({ length: p }, (__, j) => (i === j ? 1 : 0)),
    );
  }

  if (!skipRefinement && bestTheta) {
    const { Theta: refinedTheta } = runGlasso(
      R, bestLambda, 10000, 1e-8, 500, 1e-10,
      bestW ?? undefined,
      bestBeta ?? undefined,
    );
    return { Theta: refinedTheta, lambda: bestLambda };
  }

  return { Theta: bestTheta!, lambda: bestLambda };
}
