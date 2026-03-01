/**
 * Precision matrix -> partial correlation conversion.
 */

export function thetaToPcor(Theta: number[][]): number[][] {
  const p = Theta.length;
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
