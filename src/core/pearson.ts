/**
 * Pearson correlation matrix (matches R cor() exactly).
 */

export function computePearsonMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  const p = matrix[0]?.length ?? 0;
  if (n < 2 || p === 0) return Array.from({ length: p }, (_, i) => Array.from({ length: p }, (__, j) => i === j ? 1 : 0));

  // Column means
  const means = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) means[j]! += matrix[i]![j]!;
  for (let j = 0; j < p; j++) means[j]! /= n;

  // Column standard deviations (sample, ddof=1 to match R)
  const sds = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      const d = matrix[i]![j]! - means[j]!;
      sds[j]! += d * d;
    }
  }
  for (let j = 0; j < p; j++) sds[j]! = Math.sqrt(sds[j]! / (n - 1));

  // Pearson r = cov(xi, xj) / (si * sj)
  const R: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let a = 0; a < p; a++) {
    R[a]![a] = 1;
    for (let b = a + 1; b < p; b++) {
      let cov = 0;
      for (let i = 0; i < n; i++) {
        cov += (matrix[i]![a]! - means[a]!) * (matrix[i]![b]! - means[b]!);
      }
      cov /= (n - 1);
      const denom = sds[a]! * sds[b]!;
      const r = denom < 1e-12 ? 0 : cov / denom;
      R[a]![b] = r;
      R[b]![a] = r;
    }
  }
  return R;
}
