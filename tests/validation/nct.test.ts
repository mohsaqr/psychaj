import { describe, it, expect } from 'vitest';
import { networkComparisonTest, computeNctLambdaPath } from '../../src/validation/nct';

function makeSyntheticData(n: number, p: number, seed: number): number[][] {
  let s = seed;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  return Array.from({ length: n }, () =>
    Array.from({ length: p }, () => next() * 10 - 5),
  );
}

describe('networkComparisonTest', () => {
  it('returns valid result structure', () => {
    const xData = makeSyntheticData(30, 4, 42);
    const yData = makeSyntheticData(30, 4, 99);
    const result = networkComparisonTest(xData, yData, {
      method: 'cor',
      iter: 10,
      seed: 42,
    });
    expect(result.obs_diff.length).toBe(4);
    expect(result.p_values.length).toBe(4);
    expect(result.effect_size.length).toBe(4);
    expect(result.net_x.length).toBe(4);
    expect(result.net_y.length).toBe(4);
    expect(result.nodeNames.length).toBe(4);
    expect(result.nX).toBe(30);
    expect(result.nY).toBe(30);
  });

  it('deterministic with same seed', () => {
    const xData = makeSyntheticData(20, 3, 42);
    const yData = makeSyntheticData(20, 3, 99);
    const r1 = networkComparisonTest(xData, yData, { method: 'cor', iter: 5, seed: 42 });
    const r2 = networkComparisonTest(xData, yData, { method: 'cor', iter: 5, seed: 42 });
    expect(r1.p_values).toEqual(r2.p_values);
  });

  it('p-values in [0, 1]', () => {
    const xData = makeSyntheticData(20, 3, 42);
    const yData = makeSyntheticData(20, 3, 99);
    const result = networkComparisonTest(xData, yData, { method: 'cor', iter: 10, seed: 42 });
    for (const row of result.p_values) {
      for (const p of row) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    }
  });

  it('throws on mismatched columns', () => {
    const xData = makeSyntheticData(20, 3, 42);
    const yData = makeSyntheticData(20, 4, 99);
    expect(() => networkComparisonTest(xData, yData)).toThrow('columns');
  });

  it('throws on too few observations', () => {
    const xData = makeSyntheticData(2, 3, 42);
    const yData = makeSyntheticData(20, 3, 99);
    expect(() => networkComparisonTest(xData, yData)).toThrow('at least 3');
  });
});

describe('computeNctLambdaPath', () => {
  it('returns log-spaced lambda path', () => {
    const data = makeSyntheticData(50, 4, 42);
    const lambdas = computeNctLambdaPath(data, 20);
    expect(lambdas.length).toBe(20);
    // Monotonically decreasing
    for (let i = 1; i < lambdas.length; i++) {
      expect(lambdas[i]!).toBeLessThan(lambdas[i - 1]!);
    }
  });
});
