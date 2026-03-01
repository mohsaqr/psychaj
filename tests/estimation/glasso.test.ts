import { describe, it, expect } from 'vitest';
import { runGlasso } from '../../src/estimation/glasso';
import { ebicGlasso } from '../../src/estimation/ebic-glasso';
import { thetaToPcor } from '../../src/estimation/theta-to-pcor';
import { computePearsonMatrix } from '../../src/core/pearson';

function makeSyntheticData(n: number, p: number, seed: number): number[][] {
  let s = seed;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  return Array.from({ length: n }, () =>
    Array.from({ length: p }, () => next() * 10 - 5),
  );
}

describe('runGlasso', () => {
  it('returns symmetric Theta', () => {
    const data = makeSyntheticData(50, 4, 42);
    const R = computePearsonMatrix(data);
    const { Theta } = runGlasso(R, 0.1);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(Theta[i]![j]).toBeCloseTo(Theta[j]![i]!, 10);
      }
    }
  });

  it('high penalty produces sparse off-diagonals', () => {
    const data = makeSyntheticData(50, 4, 42);
    const R = computePearsonMatrix(data);
    const { Theta } = runGlasso(R, 0.9);
    let offDiagNonzero = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (i !== j && Math.abs(Theta[i]![j]!) > 1e-8) offDiagNonzero++;
      }
    }
    // Most off-diagonal should be shrunk to zero
    expect(offDiagNonzero).toBeLessThan(12);
  });

  it('Theta diagonal is positive', () => {
    const data = makeSyntheticData(50, 4, 42);
    const R = computePearsonMatrix(data);
    const { Theta } = runGlasso(R, 0.1);
    for (let i = 0; i < 4; i++) {
      expect(Theta[i]![i]!).toBeGreaterThan(0);
    }
  });
});

describe('ebicGlasso', () => {
  it('returns Theta and lambda', () => {
    const data = makeSyntheticData(50, 4, 42);
    const R = computePearsonMatrix(data);
    const { Theta, lambda } = ebicGlasso(R, 50);
    expect(Theta.length).toBe(4);
    expect(lambda).toBeGreaterThan(0);
  });

  it('lambda decreases with gamma=0', () => {
    const data = makeSyntheticData(100, 5, 42);
    const R = computePearsonMatrix(data);
    const r1 = ebicGlasso(R, 100, 0.5);
    const r2 = ebicGlasso(R, 100, 0);
    // Smaller gamma → less penalization → smaller (or equal) lambda
    expect(r2.lambda).toBeLessThanOrEqual(r1.lambda + 1e-6);
  });
});

describe('thetaToPcor', () => {
  it('diagonal is 0', () => {
    const Theta = [[2, -0.5], [-0.5, 3]];
    const pcor = thetaToPcor(Theta);
    expect(pcor[0]![0]).toBe(0);
    expect(pcor[1]![1]).toBe(0);
  });

  it('values bounded in [-1, 1]', () => {
    const Theta = [[2, -0.5, 0.1], [-0.5, 3, -0.3], [0.1, -0.3, 1.5]];
    const pcor = thetaToPcor(Theta);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(Math.abs(pcor[i]![j]!)).toBeLessThanOrEqual(1 + 1e-10);
      }
    }
  });

  it('symmetric', () => {
    const Theta = [[2, -0.5, 0.1], [-0.5, 3, -0.3], [0.1, -0.3, 1.5]];
    const pcor = thetaToPcor(Theta);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(pcor[i]![j]).toBeCloseTo(pcor[j]![i]!, 10);
      }
    }
  });
});
