import { describe, it, expect } from 'vitest';
import { bootstrapGlasso, edgeKey, parseEdgeKey } from '../../src/validation/bootstrap';

function makeSyntheticData(n: number, p: number, seed: number): number[][] {
  let s = seed;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  return Array.from({ length: n }, () =>
    Array.from({ length: p }, () => next() * 10 - 5),
  );
}

describe('bootstrapGlasso', () => {
  it('returns edges with CIs', () => {
    const data = makeSyntheticData(50, 3, 42);
    const labels = ['A', 'B', 'C'];
    const result = bootstrapGlasso(data, labels, {
      method: 'cor',
      iter: 20,
      seed: 42,
    });
    // 3 variables → 3 edges
    expect(result.edges.length).toBe(3);
    for (const e of result.edges) {
      expect(e.ciLower).toBeLessThanOrEqual(e.ciUpper);
      expect(typeof e.bootMean).toBe('number');
      expect(typeof e.bootSd).toBe('number');
      expect(e.bootSamples.length).toBe(20);
    }
  });

  it('pairwise diffs computed', () => {
    const data = makeSyntheticData(50, 3, 42);
    const labels = ['A', 'B', 'C'];
    const result = bootstrapGlasso(data, labels, {
      method: 'cor',
      iter: 20,
      seed: 42,
    });
    // C(3,2) = 3 pairwise diffs
    expect(result.pairwiseDiffs.length).toBe(3);
  });

  it('returns empty for too few data', () => {
    const data = makeSyntheticData(3, 3, 42);
    const labels = ['A', 'B', 'C'];
    const result = bootstrapGlasso(data, labels, { method: 'cor', iter: 5 });
    expect(result.edges.length).toBe(0);
    expect(result.pairwiseDiffs.length).toBe(0);
  });

  it('deterministic with same seed', () => {
    const data = makeSyntheticData(40, 3, 42);
    const labels = ['A', 'B', 'C'];
    const r1 = bootstrapGlasso(data, labels, { method: 'cor', iter: 10, seed: 42 });
    const r2 = bootstrapGlasso(data, labels, { method: 'cor', iter: 10, seed: 42 });
    expect(r1.edges[0]!.bootMean).toBe(r2.edges[0]!.bootMean);
  });
});

describe('edgeKey / parseEdgeKey', () => {
  it('round-trip', () => {
    const key = edgeKey('A', 'B');
    expect(key).toBe('A||B');
    const [from, to] = parseEdgeKey(key);
    expect(from).toBe('A');
    expect(to).toBe('B');
  });
});
