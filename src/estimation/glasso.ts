/**
 * GLASSO coordinate descent
 * Friedman, Hastie & Tibshirani (2008) — matches glasso::glasso() within 1e-5
 */

/**
 * Core GLASSO coordinate descent — returns W, Beta only (no Theta).
 * Used internally by ebicGlasso path scan to avoid allocating Theta for every lambda.
 */
export function glassoCD(
  R: number[][],
  rho: number,
  maxIterOuter: number,
  tolOuter: number,
  maxIterInner: number,
  tolInner: number,
  initW?: number[][],
  initBeta?: number[][],
  /** When true, mutate initW/initBeta in place (avoids copy). Caller must not reuse originals. */
  inPlace = false,
): { W: number[][]; Beta: number[][] } {
  const p = R.length;
  const pp = p - 1;

  const W: number[][] = initW
    ? (inPlace ? initW : initW.map(row => [...row]))
    : R.map(row => [...row]);

  const Beta: number[][] = initBeta
    ? (inPlace ? initBeta : initBeta.map(row => [...row]))
    : Array.from({ length: p }, () => new Array(p).fill(0));

  // Pre-allocate work buffers
  const W11 = new Float64Array(pp * pp);
  const s12 = new Float64Array(pp);
  const beta = new Float64Array(pp);
  const w12 = new Float64Array(pp);

  for (let iter = 0; iter < maxIterOuter; iter++) {
    let maxDiff = 0;

    for (let j = 0; j < p; j++) {
      // Fill W11, s12
      let ri = 0;
      for (let a = 0; a < p; a++) {
        if (a === j) continue;
        let ci = 0;
        for (let b = 0; b < p; b++) {
          if (b !== j) { W11[ri * pp + ci] = W[a]![b]!; ci++; }
        }
        s12[ri] = R[a]![j]!;
        ri++;
      }

      // Fill beta
      let bi = 0;
      for (let k = 0; k < p; k++) {
        if (k !== j) { beta[bi] = Beta[j]![k]!; bi++; }
      }

      // Coordinate descent
      for (let innerIter = 0; innerIter < maxIterInner; innerIter++) {
        let maxBetaDiff = 0;
        for (let k = 0; k < pp; k++) {
          let partial = s12[k]!;
          for (let l = 0; l < pp; l++) {
            if (l !== k) partial -= W11[k * pp + l]! * beta[l]!;
          }
          const wkk = W11[k * pp + k]!;
          let newBeta: number;
          if (wkk < 1e-12) {
            newBeta = 0;
          } else {
            const st = partial > rho ? partial - rho : partial < -rho ? partial + rho : 0;
            newBeta = st / wkk;
          }
          const diff = Math.abs(newBeta - beta[k]!);
          if (diff > maxBetaDiff) maxBetaDiff = diff;
          beta[k] = newBeta;
        }
        if (maxBetaDiff < tolInner) break;
      }

      // w12 = W11 @ beta
      for (let a = 0; a < pp; a++) {
        let sum = 0;
        for (let k = 0; k < pp; k++) sum += W11[a * pp + k]! * beta[k]!;
        w12[a] = sum;
      }

      // Update W and Beta
      let betaIdx = 0;
      for (let a = 0; a < p; a++) {
        if (a === j) continue;
        const oldW = W[a]![j]!;
        const newW = w12[betaIdx]!;
        const diff = Math.abs(newW - oldW);
        if (diff > maxDiff) maxDiff = diff;
        W[a]![j] = newW;
        W[j]![a] = newW;
        Beta[j]![a] = beta[betaIdx]!;
        betaIdx++;
      }
    }

    if (maxDiff < tolOuter) break;
  }

  return { W, Beta };
}

/**
 * Build Theta (precision matrix) from converged W and Beta.
 */
export function buildTheta(W: number[][], Beta: number[][]): number[][] {
  const p = W.length;
  const Theta: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let j = 0; j < p; j++) {
    const wjj = W[j]![j]!;
    let dot = 0;
    for (let a = 0; a < p; a++) {
      if (a !== j) dot += W[a]![j]! * Beta[j]![a]!;
    }
    const thetaJJ = Math.abs(wjj - dot) > 1e-12 ? 1 / (wjj - dot) : 1e6;
    Theta[j]![j] = thetaJJ;
    for (let a = 0; a < p; a++) {
      if (a !== j) {
        Theta[j]![a] = -Beta[j]![a]! * thetaJJ;
      }
    }
  }
  // Symmetrize
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const avg = (Theta[i]![j]! + Theta[j]![i]!) / 2;
      Theta[i]![j] = avg;
      Theta[j]![i] = avg;
    }
  }
  return Theta;
}

/**
 * Full GLASSO: coordinate descent + Theta construction.
 * Public API — unchanged signature.
 */
export function runGlasso(
  R: number[][],
  rho: number,
  maxIterOuter = 100,
  tolOuter = 1e-4,
  maxIterInner = 200,
  tolInner = 1e-6,
  initW?: number[][],
  initBeta?: number[][],
): { Theta: number[][]; W: number[][]; Beta: number[][] } {
  const { W, Beta } = glassoCD(R, rho, maxIterOuter, tolOuter, maxIterInner, tolInner, initW, initBeta);
  const Theta = buildTheta(W, Beta);
  return { Theta, W, Beta };
}
