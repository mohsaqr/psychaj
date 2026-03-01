import { describe, it, expect } from 'vitest';
import { computePageRank } from '../../src/graph/pagerank';

describe('computePageRank', () => {
  it('uniform for symmetric graph', () => {
    const weights = [
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ];
    const pr = computePageRank(weights);
    expect(pr.length).toBe(3);
    // All nodes should have equal PageRank ≈ 1/3
    for (let i = 0; i < 3; i++) {
      expect(pr[i]!).toBeCloseTo(1 / 3, 2);
    }
  });

  it('sink node gets rank', () => {
    // 0 → 1 → 2 (no outgoing from 2)
    const weights = [
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 0],
    ];
    const pr = computePageRank(weights);
    expect(pr.length).toBe(3);
    // All values should sum to ~1
    const sum = pr[0]! + pr[1]! + pr[2]!;
    expect(sum).toBeCloseTo(1, 2);
  });

  it('empty graph', () => {
    const pr = computePageRank([]);
    expect(pr.length).toBe(0);
  });

  it('values sum to 1', () => {
    const weights = [
      [0, 0.5, 0.3, 0],
      [0.2, 0, 0, 0.4],
      [0, 0.1, 0, 0.5],
      [0.3, 0, 0.2, 0],
    ];
    const pr = computePageRank(weights);
    let sum = 0;
    for (let i = 0; i < 4; i++) sum += pr[i]!;
    expect(sum).toBeCloseTo(1, 2);
  });
});
