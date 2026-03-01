/**
 * Partial correlation via Moore-Penrose pseudoinverse.
 * Matches corpcor::cor2pcor() in R.
 * Uses Jacobi eigendecomposition — handles rank-deficient R.
 */

// ═══════════════════════════════════════════════════════════
//  Jacobi eigenvalue decomposition (symmetric matrices)
// ═══════════════════════════════════════════════════════════

export function jacobiEig(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length;
  const M = A.map(row => [...row]);
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => (i === j ? 1 : 0)),
  );

  for (let sweep = 0; sweep < n * 30; sweep++) {
    let maxOff = 0, pi = 0, qi = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(M[i]![j]!);
        if (v > maxOff) { maxOff = v; pi = i; qi = j; }
      }
    }
    if (maxOff < 1e-14) break;

    const mpq = M[pi]![qi]!;
    const mpp = M[pi]![pi]!;
    const mqq = M[qi]![qi]!;
    const tau = (mqq - mpp) / (2 * mpq);
    const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    M[pi]![pi] = mpp - t * mpq;
    M[qi]![qi] = mqq + t * mpq;
    M[pi]![qi] = 0; M[qi]![pi] = 0;

    for (let r = 0; r < n; r++) {
      if (r !== pi && r !== qi) {
        const mrp = M[r]![pi]!; const mrq = M[r]![qi]!;
        const nrp = c * mrp - s * mrq; const nrq = s * mrp + c * mrq;
        M[r]![pi] = nrp; M[pi]![r] = nrp;
        M[r]![qi] = nrq; M[qi]![r] = nrq;
      }
    }
    for (let r = 0; r < n; r++) {
      const vrp = V[r]![pi]!; const vrq = V[r]![qi]!;
      V[r]![pi] = c * vrp - s * vrq;
      V[r]![qi] = s * vrp + c * vrq;
    }
  }

  return { values: M.map((row, i) => row[i]!), vectors: V };
}

// ═══════════════════════════════════════════════════════════
//  Moore-Penrose pseudoinverse via Jacobi eigendecomposition
// ═══════════════════════════════════════════════════════════

export function pseudoInverseSym(A: number[][]): number[][] {
  const n = A.length;
  const { values, vectors } = jacobiEig(A);

  const maxAbs = Math.max(...values.map(Math.abs), 1e-15);
  const tol = Math.max(n * Number.EPSILON * maxAbs * 10, maxAbs * 1e-4);
  const dplus = values.map(v => Math.abs(v) > tol ? 1 / v : 0);

  const result: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += vectors[i]![k]! * dplus[k]! * vectors[j]![k]!;
      result[i]![j] = sum;
      result[j]![i] = sum;
    }
  }
  return result;
}

// ═══════════════════════════════════════════════════════════
//  Partial correlation (matches corpcor::cor2pcor())
// ═══════════════════════════════════════════════════════════

export function computePartialCorrMatrix(R: number[][]): number[][] {
  const p = R.length;
  const Theta = pseudoInverseSym(R);

  const pcor: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      if (i === j) continue;
      const denom = Math.sqrt(Theta[i]![i]! * Theta[j]![j]!);
      pcor[i]![j] = denom < 1e-12 ? 0 : -Theta[i]![j]! / denom;
    }
  }
  return pcor;
}
