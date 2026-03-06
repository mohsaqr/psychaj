/**
 * IsingFit — L1-regularized Ising model estimation
 *
 * Estimates a binary network via node-wise L1-penalized logistic regression
 * with EBIC model selection (van Borkulo et al., 2014).
 *
 * Each node is regressed on all other nodes using binomial glmnet.
 * Lambda is selected per-node via EBIC. The asymmetric coefficient matrix
 * is symmetrized via AND or OR rule.
 */

import { glmnetPathFlat, selectByEBIC } from 'carm/stats';
import type { IsingFitOptions, IsingFitResult } from '../core/types';

/**
 * Fit an Ising model to binary (0/1) data.
 *
 * @param data  n×p matrix (array of n rows, each of p binary values)
 * @param labels  p variable names
 * @param opts  options (gamma, rule, nLambda)
 * @returns  IsingFitResult with symmetric weight matrix and thresholds
 */
export function fitIsing(
  data: number[][],
  labels: string[],
  opts?: IsingFitOptions,
): IsingFitResult {
  const gamma = opts?.gamma ?? 0.25;
  const rule = opts?.rule ?? 'AND';
  const nLambda = opts?.nLambda ?? 100;

  const n = data.length;
  const p = labels.length;

  if (n === 0) throw new Error('fitIsing: data must be non-empty');
  if (p < 2) throw new Error('fitIsing: need at least 2 variables');
  if ((data[0]?.length ?? 0) !== p) throw new Error('fitIsing: data columns must match labels length');

  // Validate binary (0/1) and sufficient variance
  for (let j = 0; j < p; j++) {
    let nOnes = 0;
    for (let i = 0; i < n; i++) {
      const v = data[i]![j]!;
      if (v !== 0 && v !== 1) throw new Error(`fitIsing: column ${j} ("${labels[j]}") contains non-binary value ${v}`);
      nOnes += v;
    }
    if (nOnes === 0 || nOnes === n) {
      throw new Error(`fitIsing: column ${j} ("${labels[j]}") has zero variance`);
    }
  }

  // ── Pre-flatten data into Float64Array for efficient node-wise regression ──
  const dataFlat = new Float64Array(n * p);
  for (let i = 0; i < n; i++) {
    const row = data[i]!;
    for (let j = 0; j < p; j++) dataFlat[i * p + j] = row[j]!;
  }

  // ── Node-wise regression ──
  // asymWeights[s][t] = coefficient of node t in regression of node s
  const asymWeights: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const thresholds: number[] = new Array(p).fill(0);
  const selectedLambdas: number[] = new Array(p).fill(0);
  const pMinus1 = p - 1;

  for (let s = 0; s < p; s++) {
    // Build flat sub-matrix: y = column s, X = all other columns
    const Xflat = new Float64Array(n * pMinus1);
    const yFlat = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      yFlat[i] = dataFlat[i * p + s]!;
      let ci = 0;
      for (let j = 0; j < p; j++) {
        if (j !== s) { Xflat[i * pMinus1 + ci] = dataFlat[i * p + j]!; ci++; }
      }
    }

    const path = glmnetPathFlat(Xflat, yFlat, n, pMinus1, {
      family: 'binomial',
      nLambda,
      standardize: true,  // match R's IsingFit which uses glmnet default standardize=TRUE
      intercept: true,
    });

    const best = selectByEBIC(path, gamma);
    thresholds[s] = best.coefs.intercept;
    selectedLambdas[s] = best.coefs.lambda;

    // Map coefficients back to original node indices
    let ci = 0;
    for (let j = 0; j < p; j++) {
      if (j !== s) {
        asymWeights[s]![j] = best.coefs.beta[ci]!;
        ci++;
      }
    }
  }

  // ── Symmetrize ──
  const weightMatrix: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));

  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const wij = asymWeights[i]![j]!;
      const wji = asymWeights[j]![i]!;

      let sym: number;
      if (rule === 'AND') {
        // Both must be nonzero
        if (Math.abs(wij) > 1e-10 && Math.abs(wji) > 1e-10) {
          sym = (wij + wji) / 2;
        } else {
          sym = 0;
        }
      } else {
        // OR: average regardless
        sym = (wij + wji) / 2;
      }

      weightMatrix[i]![j] = sym;
      weightMatrix[j]![i] = sym;
    }
  }

  return {
    weightMatrix,
    thresholds,
    labels: [...labels],
    gamma,
    rule,
    lambdas: selectedLambdas,
    nObs: n,
  };
}
