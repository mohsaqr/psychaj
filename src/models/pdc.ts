/**
 * Partial Directed Correlations (PDC).
 *
 * PDC[k][j] = beta[j][k] / sqrt(sigma[k][k] * kappa[j][j] + beta[j][k]²)
 * where sigma = kappa^{-1}.
 */

import { invertSymmetric } from '../core/linalg';

export function computePDC(
  beta: number[][],
  kappa: number[][],
): number[][] {
  const d = beta.length;
  const sigma = invertSymmetric(kappa);

  const PDC: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let j = 0; j < d; j++) {
    for (let k = 0; k < d; k++) {
      const b = beta[j]![k]!;
      if (Math.abs(b) < 1e-15) continue;
      const denom = Math.sqrt(sigma[k]![k]! * kappa[j]![j]! + b * b);
      PDC[j]![k] = denom > 1e-15 ? b / denom : 0;
    }
  }
  return PDC;
}
