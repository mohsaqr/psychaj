/**
 * mlVAR — Multilevel Vector Autoregression
 *
 * Produces three networks from EMA/ESM repeated-measures data:
 *   temporal        — directed (d×d), OLS fixed-effect VAR coefficients
 *   contemporaneous — undirected (d×d), EBIC-GLASSO on within-person OLS residuals
 *   between         — undirected (d×d), GGM nodewise regression on person means (K = D(I-Γ))
 *
 * References: Epskamp et al. (2018) Psychological Methods.
 */

import { multipleRegression, runLMM } from 'carm';
import { computePearsonMatrix } from '../core/pearson';
import { jacobiEig } from '../core/partial-corr';
import { ebicGlasso } from '../estimation/ebic-glasso';
import { thetaToPcor } from '../estimation/theta-to-pcor';
import { tDistCDF, tDistInv } from '../core/stats';
import type { MlVAROptions, MlVARCoef, MlVARResult, ImpulseResponse } from '../core/types';

export function fitMlVAR(
  rows: Record<string, string | number>[],
  opts: MlVAROptions,
): MlVARResult {
  const { vars, idvar, dayvar, beepvar, lag = 1, standardize = true, gamma = 0.5, computeIndividual = false } = opts;
  const d = vars.length;

  if (d < 2) throw new Error('mlVAR requires at least 2 variables.');
  if (rows.length === 0) throw new Error('No data rows provided.');

  // ── 1. Sort rows by (id, day, beep)
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

  // ── 2. Compute per-subject means
  const subjectIdx = new Map<string, number[]>();
  for (let i = 0; i < sorted.length; i++) {
    const id = String(sorted[i]![idvar] ?? '');
    if (!subjectIdx.has(id)) subjectIdx.set(id, []);
    subjectIdx.get(id)!.push(i);
  }
  const subjects = [...subjectIdx.keys()];
  const nSubjects = subjects.length;

  const personMeans = new Map<string, number[]>();
  for (const [id, indices] of subjectIdx) {
    const mean = new Array<number>(d).fill(0);
    for (const i of indices) {
      for (let j = 0; j < d; j++) mean[j]! += Number(sorted[i]![vars[j]!] ?? 0);
    }
    for (let j = 0; j < d; j++) mean[j]! /= indices.length;
    personMeans.set(id, mean);
  }

  // ── 3. Compute pooled SDs
  let sds: number[] | null = null;
  if (standardize) {
    sds = new Array<number>(d).fill(1);
    for (let j = 0; j < d; j++) {
      const vals = sorted.map(r => Number(r[vars[j]!] ?? 0));
      const mu = vals.reduce((s, v) => s + v, 0) / vals.length;
      const ss = vals.reduce((s, v) => s + (v - mu) ** 2, 0);
      const sd = Math.sqrt(ss / (vals.length - 1));
      sds[j] = sd > 1e-12 ? sd : 1;
    }
  }

  const betweenMat: number[][] = subjects.map(id => {
    const means = personMeans.get(id)!;
    return sds ? means.map((m, j) => m / sds[j]!) : means;
  });

  // ── 4. Build within-person lag pairs
  const xCentered: number[][] = [];
  const yCentered: number[][] = [];
  const pairSubjectIds: string[] = [];

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

    const mean = personMeans.get(curId)!;

    const yVec: number[] = vars.map((v, j) => {
      const c = Number(cur[v] ?? 0) - mean[j]!;
      return sds ? c / sds[j]! : c;
    });
    const xVec: number[] = vars.map((v, j) => {
      const c = Number(lgg[v] ?? 0) - mean[j]!;
      return sds ? c / sds[j]! : c;
    });

    yCentered.push(yVec);
    xCentered.push(xVec);
    pairSubjectIds.push(curId);
  }

  const nObs = xCentered.length;
  if (nObs < d + 2) {
    throw new Error(
      `Insufficient within-person lag pairs (${nObs}) for ${d} variables. ` +
      `Need at least ${d + 2}. ` +
      'Check that the ID, Day, and Beep columns are correctly set and there are enough repeated observations per person.',
    );
  }

  // ── 4b. Compute R-style lagged person means for between-subjects centering.
  // R's mlVAR expands beep gaps and computes lagged predictor person means
  // over ALL non-NA lagged values (including positions from gap expansion).
  // Each gap (beep_diff > 1) adds one extra lagged value: the observation
  // right before the gap. We compute these means for _betweenGGM centering.
  const _laggedSums = new Map<string, number[]>();
  const _laggedCounts = new Map<string, number[]>();
  const _initLagged = (id: string) => {
    if (!_laggedSums.has(id)) {
      _laggedSums.set(id, new Array(d).fill(0));
      _laggedCounts.set(id, new Array(d).fill(0));
    }
  };
  // Add all valid lag pair lagged values (raw / sd)
  for (let t = 0; t < nObs; t++) {
    const id = pairSubjectIds[t]!;
    _initLagged(id);
    const sums = _laggedSums.get(id)!;
    const counts = _laggedCounts.get(id)!;
    const pm = personMeans.get(id)!;
    for (let j = 0; j < d; j++) {
      // raw_lagged / sd = xCentered + personMean / sd
      const rawOverSd = sds ? xCentered[t]![j]! + pm[j]! / sds[j]! : xCentered[t]![j]! + pm[j]!;
      sums[j]! += rawOverSd;
      counts[j]! += 1;
    }
  }
  // Add extra lagged values from beep gaps within same (id, day)
  for (let i = 1; i < sorted.length; i++) {
    const curId = String(sorted[i]![idvar] ?? '');
    const prevId = String(sorted[i - 1]![idvar] ?? '');
    if (curId !== prevId) continue;
    if (dayvar && Number(sorted[i]![dayvar] ?? 0) !== Number(sorted[i - 1]![dayvar] ?? 0)) continue;
    const beepDiff = beepvar ? Number(sorted[i]![beepvar] ?? 0) - Number(sorted[i - 1]![beepvar] ?? 0) : 1;
    if (beepDiff <= 1) continue; // Not a gap
    // Gap of size beepDiff-1: the value at sorted[i-1] becomes an extra lagged value
    _initLagged(curId);
    const sums = _laggedSums.get(curId)!;
    const counts = _laggedCounts.get(curId)!;
    for (let j = 0; j < d; j++) {
      const rawOverSd = sds
        ? Number(sorted[i - 1]![vars[j]!] ?? 0) / sds[j]!
        : Number(sorted[i - 1]![vars[j]!] ?? 0);
      sums[j]! += rawOverSd;
      counts[j]! += 1;
    }
  }
  // Compute means
  const personLaggedMeans = new Map<string, number[]>();
  for (const [id, sums] of _laggedSums) {
    const counts = _laggedCounts.get(id)!;
    personLaggedMeans.set(id, sums.map((s, j) => counts[j]! > 0 ? s / counts[j]! : 0));
  }

  // ── 5. Within-estimator OLS per outcome variable
  const dfOLS     = nObs - d - 1;
  const dfCorrect = nObs - d - nSubjects;
  const dfEff     = dfCorrect > 1 ? dfCorrect : dfOLS;
  const scaleSE   = dfCorrect > 1 ? Math.sqrt(dfOLS / dfCorrect) : 1;
  const tCrit95   = tDistInv(0.975, dfEff);

  const Beta: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const residuals: number[][] = Array.from({ length: nObs }, () => new Array(d).fill(0));
  const allCoefs: MlVARCoef[] = [];

  const withinPreds = vars.map((v, j) => ({
    name: v,
    values: xCentered.map(r => r[j]!),
  }));

  for (let k = 0; k < d; k++) {
    const outcome = yCentered.map(r => r[k]!);

    let reg: ReturnType<typeof multipleRegression>;
    try {
      reg = multipleRegression(outcome, withinPreds);
    } catch {
      const zero = { estimate: 0, se: 0, tValue: 0, pValue: 1, ci: [0, 0] as [number, number] };
      for (let j = 0; j < d; j++) allCoefs.push({ from: vars[j]!, to: vars[k]!, ...zero });
      continue;
    }

    for (let j = 0; j < d; j++) {
      const coef = reg.coefficients[j + 1]!;
      Beta[j]![k] = coef.estimate;

      const se     = coef.se * scaleSE;
      const tVal   = se > 0 ? coef.estimate / se : 0;
      const pVal   = 2 * (1 - tDistCDF(Math.abs(tVal), dfEff));
      const ci: [number, number] = [coef.estimate - tCrit95 * se, coef.estimate + tCrit95 * se];

      allCoefs.push({
        from:     vars[j]!,
        to:       vars[k]!,
        estimate: coef.estimate,
        se,
        tValue:   tVal,
        pValue:   pVal,
        ci,
      });
    }

    for (let t = 0; t < nObs; t++) {
      residuals[t]![k] = reg.residuals[t]!;
    }
  }

  // ── 6. Contemporaneous network
  let contemporaneous: number[][];
  try {
    const R = computePearsonMatrix(residuals);
    const { Theta } = ebicGlasso(R, nObs, gamma);
    contemporaneous = thetaToPcor(Theta);
  } catch {
    contemporaneous = _zeroMatrix(d);
  }

  // ── 7. Between-subjects network (GGM nodewise regression)
  //   Two-step multilevel VAR (Epskamp et al. 2017, Eq. 3): K = D(I - Γ)
  //   For each outcome Y_k, regress non-centered Y on within-centered lagged
  //   predictors + standardized person means of all other variables.
  //   Γ[k][j] = fixed-effect coefficient of personMean_j in model for Y_k.
  //   D = diag(1/mu_SD²) from random intercept variance decomposition.
  let between: number[][];
  if (nSubjects < Math.max(d + 1, 3)) {
    between = _zeroMatrix(d);
  } else {
    try {
      between = _betweenGGM(d, nObs, pairSubjectIds, xCentered, yCentered, personMeans, sds, personLaggedMeans);
    } catch {
      between = _zeroMatrix(d);
    }
  }

  // ── 8. Per-subject temporal Betas (optional)
  let subjectIds: string[] | undefined;
  let perSubjectBetas: Map<string, number[][]> | undefined;
  let perSubjectNObs: Map<string, number> | undefined;

  if (computeIndividual) {
    subjectIds = [];
    perSubjectBetas = new Map();
    perSubjectNObs = new Map();

    const subjPairIdx = new Map<string, number[]>();
    for (let t = 0; t < pairSubjectIds.length; t++) {
      const id = pairSubjectIds[t]!;
      if (!subjPairIdx.has(id)) subjPairIdx.set(id, []);
      subjPairIdx.get(id)!.push(t);
    }

    for (const [id, idxs] of subjPairIdx) {
      subjectIds.push(id);
      const n = idxs.length;
      perSubjectNObs.set(id, n);

      if (n < d + 2) {
        perSubjectBetas.set(id, _zeroMatrix(d));
        continue;
      }

      const sx = idxs.map(i => xCentered[i]!);
      const sy = idxs.map(i => yCentered[i]!);
      const sBeta = _zeroMatrix(d);
      const sPreds = vars.map((v, j) => ({ name: v, values: sx.map(r => r[j]!) }));

      for (let k = 0; k < d; k++) {
        const outcome = sy.map(r => r[k]!);
        try {
          const reg = multipleRegression(outcome, sPreds);
          for (let j = 0; j < d; j++) {
            sBeta[j]![k] = reg.coefficients[j + 1]!.estimate;
          }
        } catch {
          // Degenerate: keep zeros
        }
      }
      perSubjectBetas.set(id, sBeta);
    }
  }

  const formatted = [
    'mlVAR (OLS)',
    `d=${d}`,
    `n=${nSubjects}`,
    `T=${nObs}`,
    `lag=${lag}`,
    standardize ? 'standardized' : 'unstandardized',
  ].join(', ');

  return {
    temporal: Beta, contemporaneous, between,
    labels: vars, coefs: allCoefs, nObs, nSubjects, formatted,
    subjectIds, perSubjectBetas, perSubjectNObs,
  };
}

/**
 * Compute impulse response for a VAR(1) temporal network.
 */
export function computeImpulseResponse(
  temporal: number[][],
  labels: string[],
  shockedVarIdx: number,
  nSteps = 20,
  magnitude = 1,
): ImpulseResponse {
  const d = labels.length;
  const v: number[] = new Array(d).fill(0);
  v[shockedVarIdx] = magnitude;

  const trajectories: number[][] = Array.from({ length: d }, () => new Array(nSteps).fill(0));
  for (let j = 0; j < d; j++) trajectories[j]![0] = v[j]!;

  for (let t = 1; t < nSteps; t++) {
    const newV: number[] = new Array(d).fill(0);
    for (let k = 0; k < d; k++) {
      for (let j = 0; j < d; j++) {
        newV[k]! += temporal[j]![k]! * v[j]!;
      }
    }
    for (let j = 0; j < d; j++) {
      v[j] = newV[j]!;
      trajectories[j]![t] = v[j]!;
    }
  }

  return {
    trajectories,
    labels,
    shockedVar: labels[shockedVarIdx] ?? '',
    spectralRadius: _spectralRadius(temporal),
  };
}

function _zeroMatrix(d: number): number[][] {
  return Array.from({ length: d }, () => new Array(d).fill(0));
}

/**
 * Between-subjects GGM via nodewise regression (Epskamp et al. 2017, Eq. 3).
 * Matches R's mlVAR two-step estimation: K = D(I - Γ).
 * Uses REML (carm's runLMM) for each outcome variable:
 *   Y_k ~ [within-centered lagged] + [person means of other vars] + (1|id)
 */
function _betweenGGM(
  d: number,
  nObs: number,
  pairSubjectIds: string[],
  xCentered: number[][],
  yCentered: number[][],
  personMeans: Map<string, number[]>,
  sds: number[] | null,
  personLaggedMeans: Map<string, number[]>,
): number[][] {
  const Gamma: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const muSD: number[] = new Array(d).fill(1);

  // Standardized person means (personMean / sd for each variable)
  const pmStd = new Map<string, number[]>();
  for (const [id, mean] of personMeans) {
    pmStd.set(id, sds ? mean.map((m, j) => m / sds[j]!) : [...mean]);
  }

  for (let k = 0; k < d; k++) {
    // Outcome: non-centered standardized Y (= yCentered + personMeanStd)
    const yRaw: number[] = new Array(nObs);
    for (let t = 0; t < nObs; t++) {
      const pm = pmStd.get(pairSubjectIds[t]!)!;
      yRaw[t] = yCentered[t]![k]! + pm[k]!;
    }

    // Build fixed predictors for runLMM
    const fixedPredictors: Record<string, number[]> = {};

    // d within-centered lagged predictors.
    // R centers by the mean of ALL lagged values (including gap-expansion positions).
    // xCentered[t][j] + pmStd[j] = raw_lagged / sd. Subtract R-style lagged mean.
    for (let j = 0; j < d; j++) {
      fixedPredictors[`w${j}`] = xCentered.map((r, t) => {
        const id = pairSubjectIds[t]!;
        const pm = pmStd.get(id)!;
        const rawOverSd = r[j]! + pm[j]!; // raw_lagged / sd
        return rawOverSd - personLaggedMeans.get(id)![j]!;
      });
    }

    // (d-1) between-subject predictors: person means of vars ≠ k
    const betweenIdx: number[] = [];
    for (let j = 0; j < d; j++) {
      if (j === k) continue;
      betweenIdx.push(j);
      fixedPredictors[`b${j}`] = pairSubjectIds.map(id => pmStd.get(id)![j]!);
    }

    try {
      const lmm = runLMM({
        outcome: yRaw,
        fixedPredictors,
        groupId: pairSubjectIds,
        method: 'ML',
      });

      // Extract between coefficients from fixed effects
      for (let idx = 0; idx < betweenIdx.length; idx++) {
        const j = betweenIdx[idx]!;
        const fe = lmm.fixedEffects.find(f => f.name === `b${j}`);
        if (fe) Gamma[k]![j] = fe.estimate;
      }

      // Random intercept SD (keep zero if variance is zero — triggers zero-network)
      muSD[k] = Math.sqrt(Math.max(lmm.varianceComponents.intercept, 0));
    } catch {
      muSD[k] = 1e-5;
    }
  }

  // R returns zero between network when any mu_SD is zero
  if (muSD.some(v => v < 1e-8)) return Array.from({ length: d }, () => new Array(d).fill(0));

  // Build Kappa = D * (I - Gamma), D = diag(1/muSD²)
  const Kappa: number[][] = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => {
      const ig = (i === j ? 1 : 0) - Gamma[i]![j]!;
      return ig / (muSD[i]! * muSD[i]!);
    }),
  );

  // Symmetrize
  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      const avg = (Kappa[i]![j]! + Kappa[j]![i]!) / 2;
      Kappa[i]![j] = avg;
      Kappa[j]![i] = avg;
    }
  }

  // Force positive definite — matches R's forcePositive():
  // adds |min_eigenvalue| + 0.001 to the diagonal
  const KappaPD = _forcePositiveDefinite(Kappa);

  // Partial correlations from precision matrix (Eq. 2)
  return thetaToPcor(KappaPD);
}

/**
 * Force a symmetric matrix to be positive definite by adding
 * |min_eigenvalue| + 0.001 to the diagonal. Matches R's forcePositive().
 */
function _forcePositiveDefinite(M: number[][]): number[][] {
  const n = M.length;
  const { values } = jacobiEig(M);

  const minVal = Math.min(...values);
  if (minVal > 0) return M;

  // Add |min_eigenvalue| + 0.001 to each diagonal element
  const shift = -minVal + 0.001;
  const result: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => M[i]![j]! + (i === j ? shift : 0)),
  );
  return result;
}

function _spectralRadius(temporal: number[][]): number {
  const d = temporal.length;
  if (d === 0) return 0;
  const norm0 = 1 / Math.sqrt(d);
  let v: number[] = Array.from({ length: d }, () => norm0);
  let rho = 0;
  for (let iter = 0; iter < 80; iter++) {
    const w: number[] = new Array(d).fill(0);
    for (let k = 0; k < d; k++) {
      for (let j = 0; j < d; j++) {
        w[k]! += temporal[j]![k]! * v[j]!;
      }
    }
    rho = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
    if (rho < 1e-15) return 0;
    v = w.map(x => x / rho);
  }
  return rho;
}
