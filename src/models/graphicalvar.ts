/**
 * graphicalVAR — Sparse VAR(1) via joint L1 regularization
 *
 * Rothman–Levina–Zhu (2010) block coordinate descent with EBIC model selection.
 */

import { runGlasso } from '../estimation/glasso';
import { invertSymmetric, logDet } from '../core/linalg';
import type { GraphicalVAROptions, GraphicalVARResult } from '../core/types';
import { computePDC } from './pdc';
import { computePCC } from './pcc';

export function fitGraphicalVAR(
  rows: Record<string, string | number>[],
  opts: GraphicalVAROptions,
): GraphicalVARResult {
  const {
    vars, idvar, dayvar, beepvar,
    lag = 1,
    gamma = 0.5,
    nLambda = 50,
    scale = true,
    centerWithin = true,
    penalizeDiagonal = true,
  } = opts;
  const d = vars.length;

  if (d < 2) throw new Error('graphicalVAR requires at least 2 variables.');
  if (rows.length === 0) throw new Error('No data rows provided.');

  // ── 1. Sort rows
  const sorted = [...rows].sort((a, b) => {
    const idA = String(a[idvar] ?? '');
    const idB = String(b[idvar] ?? '');
    if (idA !== idB) return idA < idB ? -1 : 1;
    if (dayvar) {
      const da = Number(a[dayvar] ?? 0);
      const db = Number(b[dayvar] ?? 0);
      if (da !== db) return da - db;
    }
    if (beepvar) {
      return Number(a[beepvar] ?? 0) - Number(b[beepvar] ?? 0);
    }
    return 0;
  });

  // ── 2. Extract raw numeric matrix
  const rawMatrix: number[][] = sorted.map(row =>
    vars.map(v => Number(row[v] ?? 0)),
  );

  // ── 3. Compute global means and SDs
  const globalMeans = new Array<number>(d).fill(0);
  const n = rawMatrix.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) globalMeans[j]! += rawMatrix[i]![j]!;
  }
  for (let j = 0; j < d; j++) globalMeans[j]! /= n;

  let globalSDs: number[] | null = null;
  if (scale) {
    globalSDs = new Array<number>(d).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        globalSDs[j]! += (rawMatrix[i]![j]! - globalMeans[j]!) ** 2;
      }
    }
    for (let j = 0; j < d; j++) {
      globalSDs[j] = Math.sqrt(globalSDs[j]! / (n - 1));
      if (globalSDs[j]! < 1e-12) globalSDs[j] = 1;
    }
  }

  // ── 4. Scale globally
  const scaled: number[][] = rawMatrix.map(row =>
    row.map((v, j) => {
      let out = v - globalMeans[j]!;
      if (globalSDs) out /= globalSDs[j]!;
      return out;
    }),
  );

  // ── 5. Build per-subject index and within-centering
  const subjectIdx = new Map<string, number[]>();
  for (let i = 0; i < sorted.length; i++) {
    const id = String(sorted[i]![idvar] ?? '');
    if (!subjectIdx.has(id)) subjectIdx.set(id, []);
    subjectIdx.get(id)!.push(i);
  }
  const subjects = [...subjectIdx.keys()];
  const nSubjects = subjects.length;

  if (centerWithin && nSubjects > 1) {
    for (const [, indices] of subjectIdx) {
      const mean = new Array<number>(d).fill(0);
      for (const i of indices) {
        for (let j = 0; j < d; j++) mean[j]! += scaled[i]![j]!;
      }
      for (let j = 0; j < d; j++) mean[j]! /= indices.length;
      for (const i of indices) {
        for (let j = 0; j < d; j++) scaled[i]![j]! -= mean[j]!;
      }
    }
  }

  // ── 6. Build lag pairs
  const yRows: number[][] = [];
  const xRows: number[][] = [];

  for (let i = lag; i < sorted.length; i++) {
    const cur = sorted[i]!;
    const lgg = sorted[i - lag]!;

    const curId = String(cur[idvar] ?? '');
    const lagId = String(lgg[idvar] ?? '');
    if (curId !== lagId) continue;

    if (dayvar) {
      if (Number(cur[dayvar] ?? 0) !== Number(lgg[dayvar] ?? 0)) continue;
    }
    if (beepvar) {
      if (Number(cur[beepvar] ?? 0) - Number(lgg[beepvar] ?? 0) !== lag) continue;
    }

    yRows.push(scaled[i]!);
    xRows.push([1, ...scaled[i - lag]!]);
  }

  const nObs = yRows.length;
  if (nObs < d + 2) {
    throw new Error(
      `Insufficient lag pairs (${nObs}) for ${d} variables. ` +
      `Need at least ${d + 2}. ` +
      'Check that the ID, Day, and Beep columns are correctly set.',
    );
  }

  const Y = yRows;
  const X = xRows;

  // ── 7. Generate lambda grids
  const { lambdas_beta, lambdas_kappa } = _generateLambdas(X, Y, nLambda);

  // ── 8. Grid search
  let bestEBIC = Infinity;
  let bestBeta: number[][] | null = null;
  let bestKappa: number[][] | null = null;
  let bestLB = lambdas_beta[0]!;
  let bestLK = lambdas_kappa[0]!;

  for (const lb of lambdas_beta) {
    for (const lk of lambdas_kappa) {
      const result = _rothmana(X, Y, lb, lk, d, penalizeDiagonal, gamma);
      if (!result) continue;

      const { beta, kappa, EBIC: ebic } = result;

      if (ebic < bestEBIC || (ebic === bestEBIC && _countEdges(beta, kappa, d, penalizeDiagonal) < (bestBeta ? _countEdges(bestBeta, bestKappa!, d, penalizeDiagonal) : Infinity))) {
        bestEBIC = ebic;
        bestBeta = beta;
        bestKappa = kappa;
        bestLB = lb;
        bestLK = lk;
      }
    }
  }

  if (!bestBeta || !bestKappa) {
    bestBeta = Array.from({ length: d + 1 }, () => new Array(d).fill(0));
    bestKappa = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (__, j) => i === j ? 1 : 0),
    );
  }

  // ── 9. Compute PDC and PCC
  const temporal: number[][] = Array.from({ length: d }, (_, j) =>
    Array.from({ length: d }, (__, k) => bestBeta[j + 1]![k]!),
  );

  const PDC = computePDC(temporal, bestKappa);
  const PCC = computePCC(bestKappa);

  const formatted = [
    'graphicalVAR',
    `d=${d}`,
    `n=${nSubjects}`,
    `T=${nObs}`,
    `lag=${lag}`,
    `γ=${gamma}`,
    `λ_β=${bestLB.toFixed(4)}`,
    `λ_κ=${bestLK.toFixed(4)}`,
  ].join(', ');

  return {
    temporal,
    contemporaneous: PCC,
    PDC,
    PCC,
    beta: bestBeta,
    kappa: bestKappa,
    labels: vars,
    nObs,
    nSubjects,
    lambda_beta: bestLB,
    lambda_kappa: bestLK,
    EBIC: bestEBIC,
    formatted,
  };
}

// ═══════════════════════════════════════════════════════════
//  Block coordinate descent (Rothman et al. 2010)
// ═══════════════════════════════════════════════════════════

function _rothmana(
  X: number[][],
  Y: number[][],
  lambda_beta: number,
  lambda_kappa: number,
  d: number,
  penalizeDiagonal: boolean,
  gamma: number,
  maxIter = 100,
  convergence = 1e-4,
): { beta: number[][]; kappa: number[][]; EBIC: number } | null {
  const nObs = Y.length;
  const p = d + 1;

  let beta: number[][] = Array.from({ length: p }, () => new Array(d).fill(0));
  let kappa: number[][] = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (__, j) => i === j ? 1 : 0),
  );

  const lambdaMat: number[][] = Array.from({ length: p }, (_, j) =>
    Array.from({ length: d }, (__, k) => {
      if (j === 0) return 0;
      if (!penalizeDiagonal && j - 1 === k) return 0;
      return lambda_beta;
    }),
  );

  const S: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let t = 0; t < nObs; t++) {
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        S[i]![j]! += X[t]![i]! * X[t]![j]!;
      }
    }
  }

  const XtY: number[][] = Array.from({ length: p }, () => new Array(d).fill(0));
  for (let t = 0; t < nObs; t++) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < d; k++) {
        XtY[j]![k]! += X[t]![j]! * Y[t]![k]!;
      }
    }
  }

  const betaRidge = _betaRidge(X, Y, lambda_beta, d, nObs);
  let betaRidgeNorm = 0;
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < d; k++) {
      betaRidgeNorm += Math.abs(betaRidge[j]![k]!);
    }
  }
  if (betaRidgeNorm < 1e-10) betaRidgeNorm = 1;

  for (let iter = 0; iter < maxIter; iter++) {
    const prevBeta = beta.map(row => [...row]);

    // Kappa step
    const S_R = _residualCov(X, Y, beta, nObs);
    try {
      const { Theta } = runGlasso(S_R, lambda_kappa, 100, 1e-5, 200, 1e-7);
      kappa = Theta;
    } catch {
      kappa = Array.from({ length: d }, (_, i) =>
        Array.from({ length: d }, (__, j) => i === j ? 1 : 0),
      );
    }

    // Beta step
    const H: number[][] = Array.from({ length: p }, () => new Array(d).fill(0));
    for (let j = 0; j < p; j++) {
      for (let c = 0; c < d; c++) {
        let val = 0;
        for (let k = 0; k < d; k++) {
          val += XtY[j]![k]! * kappa[k]![c]!;
        }
        H[j]![c] = val;
      }
    }
    beta = _betaStep(kappa, beta, lambdaMat, d, nObs, S, H, convergence, betaRidgeNorm);

    let betaDiff = 0;
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < d; k++) {
        betaDiff += Math.abs(beta[j]![k]! - prevBeta[j]![k]!);
      }
    }

    if (betaDiff < convergence * betaRidgeNorm) break;
  }

  // ── Compute EBIC using unpenalized (refitted) likelihood
  // R's Rothmana refits kappa with glasso(WS, rho=0, zero=ZeroIndex) for the
  // likelihood calculation only, but returns the penalized kappa to the caller.
  const WS = _residualCov(X, Y, beta, nObs);
  const BIG = 1e10;
  const zeroRho: number[][] = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (__, j) => {
      if (i === j) return 0;
      return Math.abs(kappa[i]![j]!) < 1e-10 ? BIG : 0;
    }),
  );

  let refitKappa: number[][] | null = null;
  try {
    const { Theta } = runGlasso(WS, zeroRho, 100, 1e-6);
    refitKappa = Theta;
  } catch {
    // Fall back to penalized kappa for likelihood
  }
  const kappaForLik = refitKappa ?? kappa;

  const logDetK = logDet(kappaForLik);
  if (!isFinite(logDetK)) return null;

  let trKS = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      trKS += kappaForLik[i]![j]! * WS[i]![j]!;
    }
  }

  // Edge counts (from penalized kappa, matching R's pdB/pdO)
  let nEdgesBeta = 0;
  for (let j = 1; j <= d; j++) {
    for (let k = 0; k < d; k++) {
      if (lambdaMat[j]![k]! === 0) continue; // only count penalized entries
      if (Math.abs(beta[j]![k]!) > 1e-10) nEdgesBeta++;
    }
  }
  let nEdgesKappa = 0;
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      if (Math.abs(kappa[i]![j]!) > 1e-10) nEdgesKappa++;
    }
  }

  const totalEdges = nEdgesBeta + nEdgesKappa;
  const logLik = (nObs / 2) * (logDetK - trKS);
  const EBIC = -2 * logLik
    + Math.log(nObs) * totalEdges
    + 4 * gamma * Math.log(2 * d) * totalEdges;

  // Return penalized kappa (not refitted), matching R's behavior
  return { beta, kappa, EBIC };
}

function _betaStep(
  kappa: number[][],
  beta: number[][],
  lambdaMat: number[][],
  d: number,
  nObs: number,
  S: number[][],
  H: number[][],
  convergence: number,
  betaRidgeNorm: number,
  maxIter = 100,
): number[][] {
  const p = d + 1;
  const newBeta = beta.map(row => [...row]);

  for (let iter = 0; iter < maxIter; iter++) {
    const oldBeta = newBeta.map(row => [...row]);

    for (let r = 0; r < p; r++) {
      for (let c = 0; c < d; c++) {
        let u = 0;
        for (let j = 0; j < p; j++) {
          for (let k = 0; k < d; k++) {
            u += newBeta[j]![k]! * S[r]![j]! * kappa[k]![c]!;
          }
        }

        const denom = S[r]![r]! * kappa[c]![c]!;
        if (denom < 1e-15) {
          newBeta[r]![c] = 0;
          continue;
        }

        const x = newBeta[r]![c]! + (H[r]![c]! - u) / denom;
        const threshold = nObs * lambdaMat[r]![c]! / denom;

        const absX = Math.abs(x);
        newBeta[r]![c] = absX > threshold ? Math.sign(x) * (absX - threshold) : 0;
      }
    }

    let criterium = 0;
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < d; k++) {
        criterium += Math.abs(newBeta[j]![k]! - oldBeta[j]![k]!);
      }
    }
    if (criterium <= convergence * betaRidgeNorm) break;
  }

  return newBeta;
}

function _residualCov(
  X: number[][],
  Y: number[][],
  beta: number[][],
  nObs: number,
): number[][] {
  const d = Y[0]!.length;
  const p = X[0]!.length;

  const S: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));

  for (let t = 0; t < nObs; t++) {
    const resid = new Array<number>(d);
    for (let k = 0; k < d; k++) {
      let pred = 0;
      for (let j = 0; j < p; j++) {
        pred += X[t]![j]! * beta[j]![k]!;
      }
      resid[k] = Y[t]![k]! - pred;
    }

    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        S[i]![j]! += resid[i]! * resid[j]!;
      }
    }
  }

  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      S[i]![j]! /= nObs;
    }
  }

  return S;
}

function _generateLambdas(
  X: number[][],
  Y: number[][],
  nLambda: number,
): { lambdas_beta: number[]; lambdas_kappa: number[] } {
  const nObs = Y.length;
  const d = Y[0]!.length;

  const S0: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let t = 0; t < nObs; t++) {
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        S0[i]![j]! += Y[t]![i]! * Y[t]![j]!;
      }
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      S0[i]![j]! /= nObs;
    }
  }

  const corY: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    corY[i]![i] = 1;
    for (let j = i + 1; j < d; j++) {
      const denom = Math.sqrt(S0[i]![i]! * S0[j]![j]!);
      const r = denom > 1e-15 ? S0[i]![j]! / denom : 0;
      corY[i]![j] = r;
      corY[j]![i] = r;
    }
  }

  let lamKMax = 0;
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      const v = Math.abs(corY[i]![j]!);
      if (v > lamKMax) lamKMax = v;
    }
  }
  if (lamKMax < 1e-10) lamKMax = 1;

  const p = X[0]!.length;
  const XtY: number[][] = Array.from({ length: p }, () => new Array(d).fill(0));
  for (let t = 0; t < nObs; t++) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < d; k++) {
        XtY[j]![k]! += X[t]![j]! * Y[t]![k]!;
      }
    }
  }

  const YtY: number[][] = S0.map(row => row.map(v => v * nObs));
  const YtYinv = invertSymmetric(YtY);

  let lamBMax = 0;
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < d; k++) {
      let val = 0;
      for (let m = 0; m < d; m++) {
        val += XtY[j]![m]! * YtYinv[m]![k]!;
      }
      const v = Math.abs(val);
      if (v > lamBMax) lamBMax = v;
    }
  }
  if (lamBMax < 1e-10) lamBMax = 1;

  const lambdaMinRatio = 0.05;
  const makePath = (maxVal: number): number[] => {
    const logMax = Math.log(maxVal);
    const logMin = Math.log(maxVal * lambdaMinRatio);
    return Array.from({ length: nLambda }, (_, k) =>
      Math.exp(logMax + (logMin - logMax) * (k / (nLambda - 1))),
    );
  };

  return {
    lambdas_beta: makePath(lamBMax),
    lambdas_kappa: makePath(lamKMax),
  };
}

function _betaRidge(
  X: number[][],
  Y: number[][],
  lambda: number,
  d: number,
  nObs: number,
): number[][] {
  const p = X[0]!.length;
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let t = 0; t < nObs; t++) {
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        XtX[i]![j]! += X[t]![i]! * X[t]![j]!;
      }
    }
  }
  for (let i = 0; i < p; i++) {
    XtX[i]![i]! += lambda;
  }

  const XtY: number[][] = Array.from({ length: p }, () => new Array(d).fill(0));
  for (let t = 0; t < nObs; t++) {
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < d; k++) {
        XtY[j]![k]! += X[t]![j]! * Y[t]![k]!;
      }
    }
  }

  const XtXinv = invertSymmetric(XtX);
  const result: number[][] = Array.from({ length: p }, () => new Array(d).fill(0));
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < d; k++) {
      let val = 0;
      for (let m = 0; m < p; m++) {
        val += XtXinv[j]![m]! * XtY[m]![k]!;
      }
      result[j]![k] = val;
    }
  }
  return result;
}

function _countEdges(beta: number[][], kappa: number[][], d: number, penalizeDiagonal: boolean): number {
  let count = 0;
  for (let j = 1; j <= d; j++) {
    for (let k = 0; k < d; k++) {
      if (!penalizeDiagonal && j - 1 === k) continue;
      if (Math.abs(beta[j]![k]!) > 1e-10) count++;
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      if (Math.abs(kappa[i]![j]!) > 1e-10) count++;
    }
  }
  return count;
}
