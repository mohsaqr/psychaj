import { describe, it, expect } from 'vitest';
import { computeCentralities } from '../../src/graph/centrality';

describe('computeCentralities', () => {
  it('computes in/out strength', () => {
    const weights = [
      [0, 0.5, 0],
      [0.3, 0, 0.2],
      [0, 0.1, 0],
    ];
    const c = computeCentralities(weights);
    // outStrength[0] = 0.5, outStrength[1] = 0.5, outStrength[2] = 0.1
    expect(c.outStrength[0]).toBeCloseTo(0.5, 10);
    expect(c.outStrength[1]).toBeCloseTo(0.5, 10);
    expect(c.outStrength[2]).toBeCloseTo(0.1, 10);
    // inStrength[0] = 0.3, inStrength[1] = 0.6, inStrength[2] = 0.2
    expect(c.inStrength[0]).toBeCloseTo(0.3, 10);
    expect(c.inStrength[1]).toBeCloseTo(0.6, 10);
    expect(c.inStrength[2]).toBeCloseTo(0.2, 10);
  });

  it('betweenness is non-negative', () => {
    const weights = [
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [0, 0, 0, 0],
    ];
    const c = computeCentralities(weights);
    for (let i = 0; i < 4; i++) {
      expect(c.betweenness[i]!).toBeGreaterThanOrEqual(0);
    }
    // Node 1 and 2 are intermediaries
    expect(c.betweenness[1]!).toBeGreaterThan(0);
    expect(c.betweenness[2]!).toBeGreaterThan(0);
  });

  it('closeness is non-negative', () => {
    const weights = [
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ];
    const c = computeCentralities(weights, false);
    for (let i = 0; i < 3; i++) {
      expect(c.closeness[i]!).toBeGreaterThan(0);
    }
  });

  it('empty graph returns zeros', () => {
    const c = computeCentralities([]);
    expect(c.inStrength.length).toBe(0);
    expect(c.outStrength.length).toBe(0);
  });
});
