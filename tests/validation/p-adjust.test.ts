import { describe, it, expect } from 'vitest';
import { pAdjust } from '../../src/validation/p-adjust';

describe('pAdjust', () => {
  it('none returns copy', () => {
    const p = [0.01, 0.05, 0.1];
    const adj = pAdjust(p, 'none');
    expect(adj).toEqual(p);
    expect(adj).not.toBe(p); // different reference
  });

  it('bonferroni multiplies by n', () => {
    const p = [0.01, 0.05, 0.1];
    const adj = pAdjust(p, 'bonferroni');
    expect(adj[0]).toBeCloseTo(0.03, 10);
    expect(adj[1]).toBeCloseTo(0.15, 10);
    expect(adj[2]).toBeCloseTo(0.30, 10);
  });

  it('bonferroni caps at 1', () => {
    const p = [0.5, 0.6];
    const adj = pAdjust(p, 'bonferroni');
    expect(adj[0]).toBe(1);
    expect(adj[1]).toBe(1);
  });

  it('holm step-down', () => {
    const p = [0.01, 0.04, 0.1];
    const adj = pAdjust(p, 'holm');
    // Sorted: 0.01(×3), 0.04(×2), 0.1(×1) with cummax
    expect(adj[0]).toBeCloseTo(0.03, 10);
    expect(adj[1]).toBeCloseTo(0.08, 10);
    expect(adj[2]).toBeCloseTo(0.10, 10);
  });

  it('BH step-up', () => {
    const p = [0.01, 0.05, 0.1];
    const adj = pAdjust(p, 'BH');
    // Sorted: 0.01(×3/1), 0.05(×3/2), 0.1(×3/3) with cummin from top
    expect(adj[0]).toBeCloseTo(0.03, 10);
    expect(adj[1]).toBeCloseTo(0.075, 10);
    expect(adj[2]).toBeCloseTo(0.10, 10);
  });
});
