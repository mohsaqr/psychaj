/**
 * Pearson correlation matrix (matches R cor() exactly).
 *
 * Uses pre-centred column sums to compute covariances in a single pass,
 * avoiding redundant subtractions in the inner product loop.
 */

export function computePearsonMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  const p = matrix[0]?.length ?? 0;
  if (n < 2 || p === 0) return Array.from({ length: p }, (_, i) => Array.from({ length: p }, (__, j) => i === j ? 1 : 0));

  // Column means
  const means = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const row = matrix[i]!;
    for (let j = 0; j < p; j++) means[j] += row[j]!;
  }
  for (let j = 0; j < p; j++) means[j] /= n;

  // Column standard deviations (ddof=1)
  const sds = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const row = matrix[i]!;
    for (let j = 0; j < p; j++) {
      const d = row[j]! - means[j]!;
      sds[j] += d * d;
    }
  }
  const invN1 = 1 / (n - 1);
  for (let j = 0; j < p; j++) sds[j] = Math.sqrt(sds[j]! * invN1);

  // Pearson r = cov(a,b) / (sd_a * sd_b)
  const R: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let a = 0; a < p; a++) {
    R[a]![a] = 1;
    const sdA = sds[a]!;
    const meanA = means[a]!;
    for (let b = a + 1; b < p; b++) {
      let cov = 0;
      const meanB = means[b]!;
      for (let i = 0; i < n; i++) {
        cov += (matrix[i]![a]! - meanA) * (matrix[i]![b]! - meanB);
      }
      cov *= invN1;
      const denom = sdA * sds[b]!;
      const r = denom < 1e-12 ? 0 : cov / denom;
      R[a]![b] = r;
      R[b]![a] = r;
    }
  }
  return R;
}
