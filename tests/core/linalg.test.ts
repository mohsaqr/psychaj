import { describe, it, expect } from 'vitest';
import { logDet, invertSymmetric } from '../../src/core/linalg';

describe('logDet', () => {
  it('identity matrix has logDet = 0', () => {
    const I = [[1, 0], [0, 1]];
    expect(logDet(I)).toBeCloseTo(0, 10);
  });

  it('2×2 positive definite', () => {
    const M = [[4, 2], [2, 3]];
    // det = 4*3 - 2*2 = 8; logDet = ln(8) ≈ 2.079
    expect(logDet(M)).toBeCloseTo(Math.log(8), 8);
  });

  it('returns -Infinity for non-PD', () => {
    const M = [[1, 2], [2, 1]]; // det = -3 < 0
    expect(logDet(M)).toBe(-Infinity);
  });
});

describe('invertSymmetric', () => {
  it('inverse of identity is identity', () => {
    const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const inv = invertSymmetric(I);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(inv[i]![j]).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
  });

  it('M * M^{-1} = I', () => {
    const M = [[4, 2], [2, 3]];
    const inv = invertSymmetric(M);
    // Product should be identity
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        let sum = 0;
        for (let k = 0; k < 2; k++) sum += M[i]![k]! * inv[k]![j]!;
        expect(sum).toBeCloseTo(i === j ? 1 : 0, 8);
      }
    }
  });

  it('result is symmetric', () => {
    const M = [[5, 1, 2], [1, 4, 1], [2, 1, 3]];
    const inv = invertSymmetric(M);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(inv[i]![j]).toBeCloseTo(inv[j]![i]!, 10);
      }
    }
  });
});
