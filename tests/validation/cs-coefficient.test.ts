import { describe, it, expect } from 'vitest';
import { estimateCS } from '../../src/validation/cs-coefficient';
import { computePearsonMatrix } from '../../src/core/pearson';

function makeSyntheticData(n: number, p: number, seed: number): number[][] {
  let s = seed;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  return Array.from({ length: n }, () =>
    Array.from({ length: p }, () => next() * 10 - 5),
  );
}

describe('estimateCS', () => {
  it('returns valid structure', () => {
    const data = makeSyntheticData(50, 4, 42);
    const result = estimateCS(
      data,
      (subset) => computePearsonMatrix(subset),
      { iter: 10, seed: 42, measures: ['inStrength', 'outStrength'] },
    );
    expect(result.csCoefficients).toBeDefined();
    expect(result.meanCorrelations).toBeDefined();
    expect(result.dropProps.length).toBe(9);
    expect(result.threshold).toBe(0.7);
    expect(result.certainty).toBe(0.95);
  });

  it('CS coefficients are in [0, 0.9]', () => {
    const data = makeSyntheticData(50, 4, 42);
    const result = estimateCS(
      data,
      (subset) => computePearsonMatrix(subset),
      { iter: 10, seed: 42, measures: ['inStrength'] },
    );
    const cs = result.csCoefficients['inStrength']!;
    expect(cs).toBeGreaterThanOrEqual(0);
    expect(cs).toBeLessThanOrEqual(0.9);
  });

  it('handles small data gracefully', () => {
    const data = makeSyntheticData(3, 2, 42);
    const result = estimateCS(
      data,
      (subset) => computePearsonMatrix(subset),
      { iter: 5 },
    );
    // Should not throw, returns zero CS
    expect(result.csCoefficients).toBeDefined();
  });
});
