/**
 * Linear algebra utilities: Cholesky-based inversion and log-determinant.
 */

/** Cholesky-based log-determinant for a positive definite matrix. */
export function logDet(M: number[][]): number {
  const n = M.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i]![j]!;
      for (let k = 0; k < j; k++) s -= L[i]![k]! * L[j]![k]!;
      if (i === j) {
        if (s <= 0) return -Infinity;
        L[i]![j] = Math.sqrt(s);
      } else {
        L[i]![j] = L[j]![j]! < 1e-14 ? 0 : s / L[j]![j]!;
      }
    }
  }
  let logdet = 0;
  for (let i = 0; i < n; i++) logdet += 2 * Math.log(L[i]![i]!);
  return logdet;
}

/** Invert a symmetric positive-definite matrix via Cholesky factorization. */
export function invertSymmetric(M: number[][]): number[][] {
  const n = M.length;
  // Cholesky: M = L @ L^T
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i]![j]!;
      for (let k = 0; k < j; k++) s -= L[i]![k]! * L[j]![k]!;
      if (i === j) {
        L[i]![j] = Math.sqrt(Math.max(s, 1e-15));
      } else {
        L[i]![j] = L[j]![j]! < 1e-14 ? 0 : s / L[j]![j]!;
      }
    }
  }

  // Invert L (lower triangular)
  const Linv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    Linv[i]![i] = 1 / L[i]![i]!;
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = i; k < j; k++) s -= L[j]![k]! * Linv[k]![i]!;
      Linv[j]![i] = s / L[j]![j]!;
    }
  }

  // M^{-1} = L^{-T} @ L^{-1}
  const inv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = j; k < n; k++) s += Linv[k]![i]! * Linv[k]![j]!;
      inv[i]![j] = s;
      inv[j]![i] = s;
    }
  }

  return inv;
}
