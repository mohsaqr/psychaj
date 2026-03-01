import { describe, it, expect } from 'vitest';
import { computeGraphMetrics } from '../../src/graph/metrics';

describe('computeGraphMetrics', () => {
  it('complete undirected graph', () => {
    const weights = [
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ];
    const m = computeGraphMetrics(weights, false);
    expect(m.nodes).toBe(3);
    expect(m.edges).toBe(3);
    expect(m.density).toBeCloseTo(1, 10);
    expect(m.components).toBe(1);
    expect(m.largestComponentSize).toBe(3);
    expect(m.selfLoops).toBe(0);
  });

  it('directed graph edge count', () => {
    const weights = [
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 0],
    ];
    const m = computeGraphMetrics(weights, true);
    expect(m.edges).toBe(2);
    expect(m.density).toBeCloseTo(2 / 6, 10);
  });

  it('reciprocity for directed graph', () => {
    const weights = [
      [0, 1, 0],
      [1, 0, 1],
      [0, 0, 0],
    ];
    const m = computeGraphMetrics(weights, true);
    expect(m.reciprocity).not.toBeNull();
    // 3 edges total: (0,1), (1,0), (1,2). Two reciprocated: (0,1) and (1,0)
    expect(m.reciprocity).toBeCloseTo(2 / 3, 10);
  });

  it('reciprocity null for undirected', () => {
    const weights = [[0, 1], [1, 0]];
    const m = computeGraphMetrics(weights, false);
    expect(m.reciprocity).toBeNull();
  });

  it('disconnected graph has >1 components', () => {
    const weights = [
      [0, 1, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 0, 1],
      [0, 0, 1, 0],
    ];
    const m = computeGraphMetrics(weights, false);
    expect(m.components).toBe(2);
    expect(m.largestComponentSize).toBe(2);
  });

  it('self-loops counted', () => {
    const weights = [
      [0.5, 1],
      [0, 0.3],
    ];
    const m = computeGraphMetrics(weights, true);
    expect(m.selfLoops).toBe(2);
  });

  it('empty graph', () => {
    const m = computeGraphMetrics([]);
    expect(m.nodes).toBe(0);
    expect(m.edges).toBe(0);
  });
});
