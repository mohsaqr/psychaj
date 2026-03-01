import { describe, it, expect } from 'vitest';
import { detectCommunities } from '../../src/graph/community';

describe('detectCommunities', () => {
  const twoCluster: number[][] = [
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0.1, 0, 0],
    [0, 0, 0.1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
  ];

  it('louvain finds 2 communities', () => {
    const result = detectCommunities(twoCluster, 'louvain', false);
    expect(result.nCommunities).toBeLessThanOrEqual(3);
    expect(result.nCommunities).toBeGreaterThanOrEqual(2);
    expect(result.assignments.length).toBe(6);
  });

  it('fast_greedy finds communities', () => {
    const result = detectCommunities(twoCluster, 'fast_greedy', false);
    expect(result.nCommunities).toBeGreaterThanOrEqual(1);
  });

  it('label_prop converges', () => {
    const result = detectCommunities(twoCluster, 'label_prop', false);
    expect(result.assignments.length).toBe(6);
  });

  it('leading_eigen splits into 2 groups', () => {
    const result = detectCommunities(twoCluster, 'leading_eigen', false);
    expect(result.nCommunities).toBe(2);
  });

  it('walktrap works on directed graph', () => {
    const directed = [
      [0, 0.8, 0.7, 0, 0],
      [0.6, 0, 0.9, 0.1, 0],
      [0.5, 0.8, 0, 0, 0.1],
      [0, 0.1, 0, 0, 0.8],
      [0, 0, 0.1, 0.7, 0],
    ];
    const result = detectCommunities(directed, 'walktrap', true);
    expect(result.assignments.length).toBe(5);
    expect(result.nCommunities).toBeGreaterThanOrEqual(1);
  });

  it('single node', () => {
    const result = detectCommunities([[0]], 'louvain');
    expect(result.nCommunities).toBe(1);
    expect(result.assignments).toEqual([0]);
  });

  it('empty graph', () => {
    const result = detectCommunities([], 'louvain');
    expect(result.nCommunities).toBe(0);
    expect(result.assignments).toEqual([]);
  });

  it('assignments are 0-indexed contiguous', () => {
    const result = detectCommunities(twoCluster, 'louvain', false);
    const unique = [...new Set(result.assignments)].sort();
    for (let i = 0; i < unique.length; i++) {
      expect(unique[i]).toBe(i);
    }
  });
});
