/**
 * MGM — Mixed Graphical Models
 *
 * Estimates networks from mixed-type data (continuous + binary + count)
 * via node-wise L1-penalized GLM with EBIC model selection
 * (Haslbeck & Waldorp, 2020).
 *
 * Each node is regressed on all other nodes using the appropriate GLM family:
 *   - gaussian nodes → identity link (linear regression)
 *   - binary nodes   → logit link (logistic regression)
 *   - poisson nodes  → log link (Poisson regression)
 *
 * Lambda is selected per-node via EBIC. Edge weights are averaged across
 * both node regressions and symmetrized via AND or OR rule.
 */

import { glmnetPathFlat, selectByEBIC } from 'carm/stats';
import type { GlmFamily } from 'carm/stats';
import type { MgmNodeType, MgmOptions, MgmResult } from '../core/types';

const NODE_TYPE_TO_FAMILY: Record<MgmNodeType, GlmFamily> = {
  gaussian: 'gaussian',
  binary: 'binomial',
  poisson: 'poisson',
};

/**
 * Fit a Mixed Graphical Model to mixed-type data.
 *
 * @param data  n×p matrix (array of n rows, each of p values)
 * @param labels  p variable names
 * @param nodeTypes  p-length array specifying each variable's type
 * @param opts  options (gamma, rule, nLambda, scale)
 * @returns  MgmResult with symmetric weight and sign matrices
 */
export function fitMGM(
  data: number[][],
  labels: string[],
  nodeTypes: MgmNodeType[],
  opts?: MgmOptions,
): MgmResult {
  const gamma = opts?.gamma ?? 0.25;
  const rule = opts?.rule ?? 'AND';
  const nLambda = opts?.nLambda ?? 100;
  const scale = opts?.scale ?? true;

  const n = data.length;
  const p = labels.length;

  if (n === 0) throw new Error('fitMGM: data must be non-empty');
  if (p < 2) throw new Error('fitMGM: need at least 2 variables');
  if (nodeTypes.length !== p) throw new Error('fitMGM: nodeTypes length must match labels length');
  if ((data[0]?.length ?? 0) !== p) throw new Error('fitMGM: data columns must match labels length');

  // Validate data types
  for (let j = 0; j < p; j++) {
    if (nodeTypes[j] === 'binary') {
      for (let i = 0; i < n; i++) {
        const v = data[i]![j]!;
        if (v !== 0 && v !== 1) throw new Error(`fitMGM: binary column ${j} ("${labels[j]}") contains non-binary value ${v}`);
      }
    } else if (nodeTypes[j] === 'poisson') {
      for (let i = 0; i < n; i++) {
        const v = data[i]![j]!;
        if (v < 0 || v !== Math.floor(v)) throw new Error(`fitMGM: poisson column ${j} ("${labels[j]}") must be non-negative integer, got ${v}`);
      }
    }
  }

  // ── Pre-process: scale gaussian nodes ──
  // Only copy data if we need to modify it (scaling gaussian columns)
  const needsCopy = scale && nodeTypes.some(t => t === 'gaussian');
  const dataFlat = new Float64Array(n * p);
  if (needsCopy) {
    const colMean = new Float64Array(p);
    const colSd = new Float64Array(p);
    colSd.fill(1);

    // Flatten + scale in one pass
    for (let i = 0; i < n; i++) {
      const row = data[i]!;
      for (let j = 0; j < p; j++) dataFlat[i * p + j] = row[j]!;
    }
    for (let j = 0; j < p; j++) {
      if (nodeTypes[j] !== 'gaussian') continue;
      let sum = 0;
      for (let i = 0; i < n; i++) sum += dataFlat[i * p + j]!;
      colMean[j] = sum / n;

      let ss = 0;
      for (let i = 0; i < n; i++) {
        const d = dataFlat[i * p + j]! - colMean[j]!;
        ss += d * d;
      }
      const sd = Math.sqrt(ss / (n - 1));
      colSd[j] = sd > 1e-12 ? sd : 1;

      for (let i = 0; i < n; i++) {
        dataFlat[i * p + j] = (dataFlat[i * p + j]! - colMean[j]!) / colSd[j]!;
      }
    }
  } else {
    // No scaling needed — flatten original data directly
    for (let i = 0; i < n; i++) {
      const row = data[i]!;
      for (let j = 0; j < p; j++) dataFlat[i * p + j] = row[j]!;
    }
  }

  // ── Node-wise regression ──
  // asymWeights[s][t] = coefficient of node t in regression of node s
  const asymWeights: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const selectedLambdas: number[] = new Array(p).fill(0);
  const pMinus1 = p - 1;

  for (let s = 0; s < p; s++) {
    const family = NODE_TYPE_TO_FAMILY[nodeTypes[s]!]!;

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
      family,
      nLambda,
      standardize: true,  // match R's mgm which uses glmnet default standardize=TRUE
      intercept: true,
    });

    const best = selectByEBIC(path, gamma);
    selectedLambdas[s] = best.coefs.lambda;

    // Map coefficients back to original node indices
    // For binary nodes, scale by 0.5 to match R mgm convention:
    // mgm uses multinomial logistic for categorical variables, which
    // halves the interaction parameters relative to binomial logistic.
    const coefScale = nodeTypes[s] === 'binary' ? 0.5 : 1;
    let ci = 0;
    for (let j = 0; j < p; j++) {
      if (j !== s) {
        asymWeights[s]![j] = best.coefs.beta[ci]! * coefScale;
        ci++;
      }
    }
  }

  // ── Symmetrize: edge weights and signs ──
  const weightMatrix: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const signMatrix: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));

  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const wij = asymWeights[i]![j]!;
      const wji = asymWeights[j]![i]!;

      let weight: number;
      if (rule === 'AND') {
        if (Math.abs(wij) > 1e-10 && Math.abs(wji) > 1e-10) {
          weight = (Math.abs(wij) + Math.abs(wji)) / 2;
        } else {
          weight = 0;
        }
      } else {
        weight = (Math.abs(wij) + Math.abs(wji)) / 2;
      }

      weightMatrix[i]![j] = weight;
      weightMatrix[j]![i] = weight;

      // Sign: only assigned for edges between non-categorical nodes (R's mgm convention).
      // Edges involving binary (categorical) nodes get sign=0.
      if (weight > 1e-10 && nodeTypes[i] !== 'binary' && nodeTypes[j] !== 'binary') {
        const meanCoef = (wij + wji) / 2;
        signMatrix[i]![j] = meanCoef > 0 ? 1 : meanCoef < 0 ? -1 : 0;
        signMatrix[j]![i] = signMatrix[i]![j]!;
      }
    }
  }

  return {
    weightMatrix,
    signMatrix,
    labels: [...labels],
    nodeTypes: [...nodeTypes],
    gamma,
    rule,
    lambdas: selectedLambdas,
    nObs: n,
  };
}
