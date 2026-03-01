import { describe, it, expect } from 'vitest';
import { computePartialCorrMatrix, jacobiEig } from '../../src/core/partial-corr';

describe('jacobiEig', () => {
  it('eigenvalues of identity are all 1', () => {
    const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const { values } = jacobiEig(I);
    for (const v of values) expect(v).toBeCloseTo(1, 10);
  });

  it('eigenvalues of 2×2 matrix', () => {
    const A = [[2, 1], [1, 2]];
    const { values } = jacobiEig(A);
    values.sort((a, b) => a - b);
    expect(values[0]).toBeCloseTo(1, 10);
    expect(values[1]).toBeCloseTo(3, 10);
  });
});

describe('computePartialCorrMatrix', () => {
  it('diagonal is 0', () => {
    const R = [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]];
    const pcor = computePartialCorrMatrix(R);
    for (let i = 0; i < 3; i++) {
      expect(pcor[i]![i]).toBeCloseTo(0, 10);
    }
  });

  it('symmetric', () => {
    const R = [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]];
    const pcor = computePartialCorrMatrix(R);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(pcor[i]![j]).toBeCloseTo(pcor[j]![i]!, 10);
      }
    }
  });

  it('values bounded in [-1, 1]', () => {
    const R = [[1, 0.8, 0.6], [0.8, 1, 0.9], [0.6, 0.9, 1]];
    const pcor = computePartialCorrMatrix(R);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(Math.abs(pcor[i]![j]!)).toBeLessThanOrEqual(1 + 1e-10);
      }
    }
  });

  it('handles rank-deficient matrix', () => {
    // Compositional data: rows sum to 1 → rank deficient
    const data = [[0.5, 0.3, 0.2], [0.6, 0.1, 0.3], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2]];
    // Compute Pearson matrix
    const n = data.length;
    const p = 3;
    const means = new Array(p).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) means[j] += data[i]![j]!;
    for (let j = 0; j < p; j++) means[j] /= n;
    const sds = new Array(p).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) sds[j] += (data[i]![j]! - means[j]) ** 2;
    for (let j = 0; j < p; j++) sds[j] = Math.sqrt(sds[j] / (n - 1));
    const R: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let a = 0; a < p; a++) {
      R[a]![a] = 1;
      for (let b = a + 1; b < p; b++) {
        let cov = 0;
        for (let i = 0; i < n; i++) cov += (data[i]![a]! - means[a]) * (data[i]![b]! - means[b]);
        cov /= (n - 1);
        const r = cov / (sds[a] * sds[b]);
        R[a]![b] = r; R[b]![a] = r;
      }
    }
    // Should not throw
    const pcor = computePartialCorrMatrix(R);
    expect(pcor.length).toBe(3);
  });
});
