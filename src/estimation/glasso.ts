/**
 * GLASSO coordinate descent
 * Friedman, Hastie & Tibshirani (2008) — matches glasso::glasso() within 1e-5
 */

function softThreshold(x: number, lambda: number): number {
  if (x > lambda) return x - lambda;
  if (x < -lambda) return x + lambda;
  return 0;
}

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
  const p = R.length;

  const W: number[][] = initW
    ? initW.map(row => [...row])
    : R.map(row => [...row]);

  const Beta: number[][] = initBeta
    ? initBeta.map(row => [...row])
    : Array.from({ length: p }, () => new Array(p).fill(0));

  for (let iter = 0; iter < maxIterOuter; iter++) {
    let maxDiff = 0;

    for (let j = 0; j < p; j++) {
      const W11: number[][] = [];
      const s12: number[] = [];
      for (let a = 0; a < p; a++) {
        if (a === j) continue;
        const row: number[] = [];
        for (let b = 0; b < p; b++) {
          if (b !== j) row.push(W[a]![b]!);
        }
        W11.push(row);
        s12.push(R[a]![j]!);
      }

      const pp = p - 1;
      const beta = Beta[j]!.filter((_, k) => k !== j);

      for (let innerIter = 0; innerIter < maxIterInner; innerIter++) {
        let maxBetaDiff = 0;
        for (let k = 0; k < pp; k++) {
          let partial = s12[k]!;
          for (let l = 0; l < pp; l++) {
            if (l !== k) partial -= W11[k]![l]! * beta[l]!;
          }
          const wkk = W11[k]![k]!;
          const newBeta = wkk < 1e-12 ? 0 : softThreshold(partial, rho) / wkk;
          const diff = Math.abs(newBeta - beta[k]!);
          if (diff > maxBetaDiff) maxBetaDiff = diff;
          beta[k] = newBeta;
        }
        if (maxBetaDiff < tolInner) break;
      }

      const w12 = new Array<number>(pp).fill(0);
      for (let a = 0; a < pp; a++) {
        for (let k = 0; k < pp; k++) w12[a]! += W11[a]![k]! * beta[k]!;
      }

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

  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const avg = (Theta[i]![j]! + Theta[j]![i]!) / 2;
      Theta[i]![j] = avg;
      Theta[j]![i] = avg;
    }
  }

  return { Theta, W, Beta };
}
