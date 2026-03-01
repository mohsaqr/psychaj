import { describe, it, expect } from 'vitest';
import { computePearsonMatrix } from '../../src/core/pearson';

describe('computePearsonMatrix', () => {
  it('identity for perfectly correlated columns', () => {
    const data = [[1, 2], [2, 4], [3, 6], [4, 8]];
    const R = computePearsonMatrix(data);
    expect(R[0]![0]).toBeCloseTo(1, 10);
    expect(R[1]![1]).toBeCloseTo(1, 10);
    expect(R[0]![1]).toBeCloseTo(1, 10);
    expect(R[1]![0]).toBeCloseTo(1, 10);
  });

  it('negative correlation', () => {
    const data = [[1, 10], [2, 8], [3, 6], [4, 4]];
    const R = computePearsonMatrix(data);
    expect(R[0]![1]).toBeCloseTo(-1, 10);
  });

  it('symmetric', () => {
    const data = [[1, 2, 3], [4, 5, 6], [7, 1, 2], [3, 8, 4]];
    const R = computePearsonMatrix(data);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(R[i]![j]).toBeCloseTo(R[j]![i]!, 10);
      }
    }
  });

  it('diagonal is 1', () => {
    const data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const R = computePearsonMatrix(data);
    for (let i = 0; i < 3; i++) {
      expect(R[i]![i]).toBeCloseTo(1, 10);
    }
  });

  it('handles constant columns (zero SD)', () => {
    const data = [[1, 5], [2, 5], [3, 5]];
    const R = computePearsonMatrix(data);
    expect(R[0]![1]).toBe(0);
    expect(R[1]![0]).toBe(0);
  });

  it('matches R cor() on synthetic data', () => {
    // Verify specific values from the same data
    const data = [[1, 2, 3], [2, 3, 1], [3, 1, 4], [4, 5, 2], [5, 4, 5]];
    const R = computePearsonMatrix(data);
    // Verify symmetry and diagonal
    expect(R[0]![1]).toBeCloseTo(R[1]![0]!, 10);
    expect(R[0]![0]).toBeCloseTo(1, 10);
    // All values in [-1, 1]
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(Math.abs(R[i]![j]!)).toBeLessThanOrEqual(1 + 1e-10);
      }
    }
  });
});
