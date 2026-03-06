import { describe, it, expect } from 'vitest';
import { fitMGM } from '../../src/estimation/mgm';
import type { MgmNodeType } from '../../src/core/types';

// ── Deterministic data generators ──

function makeRng(seed: number): () => number {
  let s = seed;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
}

function boxMuller(rng: () => number): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makeMixedData(n: number, seed: number): {
  data: number[][];
  labels: string[];
  nodeTypes: MgmNodeType[];
} {
  // 5 nodes: 2 gaussian, 2 binary, 1 poisson
  // Gaussian nodes x1, x2 are correlated (r~0.6)
  // Binary nodes x3, x4: x3 depends on x1, x4 independent
  // Poisson node x5: depends on x2
  const rng = makeRng(seed);
  const data: number[][] = [];

  for (let i = 0; i < n; i++) {
    const z1 = boxMuller(rng);
    const z2 = 0.6 * z1 + 0.8 * boxMuller(rng);  // correlated with z1

    // x3 depends on x1
    const eta3 = 0.8 * z1;
    const p3 = 1 / (1 + Math.exp(-eta3));
    const x3 = rng() < p3 ? 1 : 0;

    // x4 independent
    const x4 = rng() < 0.5 ? 1 : 0;

    // x5 depends on x2
    const mu5 = Math.exp(0.5 + 0.3 * z2);
    // Simple Poisson via inverse CDF
    const L = Math.exp(-Math.min(20, mu5));
    let k = 0; let pp = 1;
    do { k++; pp *= rng(); } while (pp > L && k < 50);
    const x5 = k - 1;

    data.push([z1, z2, x3, x4, x5]);
  }

  return {
    data,
    labels: ['gauss1', 'gauss2', 'binary1', 'binary2', 'count1'],
    nodeTypes: ['gaussian', 'gaussian', 'binary', 'binary', 'poisson'],
  };
}

function makeAllGaussianData(n: number, p: number, seed: number): number[][] {
  const rng = makeRng(seed);
  return Array.from({ length: n }, () =>
    Array.from({ length: p }, () => boxMuller(rng)),
  );
}

function makeAllBinaryData(n: number, p: number, strength: number, seed: number): number[][] {
  const rng = makeRng(seed);
  return Array.from({ length: n }, () => {
    const row = new Array(p);
    row[0] = rng() < 0.5 ? 1 : 0;
    for (let j = 1; j < p; j++) {
      const eta = strength * (row[j - 1]! - 0.5);
      const prob = 1 / (1 + Math.exp(-eta));
      row[j] = rng() < prob ? 1 : 0;
    }
    return row;
  });
}

// ═══════════════════════════════════════════════════════════
//  Structure
// ═══════════════════════════════════════════════════════════

describe('fitMGM — structure', () => {
  const { data, labels, nodeTypes } = makeMixedData(400, 42);

  it('returns p×p symmetric weight matrix', () => {
    const result = fitMGM(data, labels, nodeTypes);
    const p = labels.length;
    expect(result.weightMatrix.length).toBe(p);
    for (let i = 0; i < p; i++) {
      expect(result.weightMatrix[i]!.length).toBe(p);
      for (let j = 0; j < p; j++) {
        expect(result.weightMatrix[i]![j]).toBeCloseTo(result.weightMatrix[j]![i]!, 10);
      }
    }
  });

  it('has zero diagonal', () => {
    const result = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < labels.length; i++) {
      expect(result.weightMatrix[i]![i]).toBe(0);
    }
  });

  it('weight matrix values are non-negative', () => {
    const result = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < labels.length; i++) {
      for (let j = 0; j < labels.length; j++) {
        expect(result.weightMatrix[i]![j]).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('sign matrix values are in {-1, 0, +1}', () => {
    const result = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < labels.length; i++) {
      for (let j = 0; j < labels.length; j++) {
        expect([-1, 0, 1]).toContain(result.signMatrix[i]![j]);
      }
    }
  });

  it('sign matrix is symmetric', () => {
    const result = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < labels.length; i++) {
      for (let j = 0; j < labels.length; j++) {
        expect(result.signMatrix[i]![j]).toBe(result.signMatrix[j]![i]);
      }
    }
  });

  it('returns correct metadata', () => {
    const result = fitMGM(data, labels, nodeTypes, { gamma: 0.5, rule: 'OR' });
    expect(result.labels).toEqual(labels);
    expect(result.nodeTypes).toEqual(nodeTypes);
    expect(result.gamma).toBe(0.5);
    expect(result.rule).toBe('OR');
    expect(result.nObs).toBe(400);
    expect(result.lambdas.length).toBe(5);
  });
});

// ═══════════════════════════════════════════════════════════
//  Edge detection
// ═══════════════════════════════════════════════════════════

describe('fitMGM — edge detection', () => {
  const { data, labels, nodeTypes } = makeMixedData(500, 42);

  it('detects gauss1-gauss2 edge (correlated)', () => {
    const result = fitMGM(data, labels, nodeTypes, { rule: 'OR' });
    expect(result.weightMatrix[0]![1]).toBeGreaterThan(0);
  });

  it('binary2 (independent) has fewer edges than binary1', () => {
    const result = fitMGM(data, labels, nodeTypes, { rule: 'OR' });
    let edges3 = 0; // binary1
    let edges4 = 0; // binary2
    for (let j = 0; j < 5; j++) {
      if (j !== 2 && result.weightMatrix[2]![j]! > 1e-10) edges3++;
      if (j !== 3 && result.weightMatrix[3]![j]! > 1e-10) edges4++;
    }
    expect(edges4).toBeLessThanOrEqual(edges3);
  });
});

// ═══════════════════════════════════════════════════════════
//  AND vs OR rule
// ═══════════════════════════════════════════════════════════

describe('fitMGM — AND vs OR', () => {
  const { data, labels, nodeTypes } = makeMixedData(300, 42);

  it('AND is sparser than or equal to OR', () => {
    const andResult = fitMGM(data, labels, nodeTypes, { rule: 'AND' });
    const orResult = fitMGM(data, labels, nodeTypes, { rule: 'OR' });

    let andEdges = 0;
    let orEdges = 0;
    const p = labels.length;
    for (let i = 0; i < p; i++) {
      for (let j = i + 1; j < p; j++) {
        if (andResult.weightMatrix[i]![j]! > 1e-10) andEdges++;
        if (orResult.weightMatrix[i]![j]! > 1e-10) orEdges++;
      }
    }
    expect(andEdges).toBeLessThanOrEqual(orEdges);
  });
});

// ═══════════════════════════════════════════════════════════
//  All-gaussian behaves like neighborhood selection
// ═══════════════════════════════════════════════════════════

describe('fitMGM — all-gaussian', () => {
  const data = makeAllGaussianData(200, 4, 42);
  const labels = ['A', 'B', 'C', 'D'];
  const nodeTypes: MgmNodeType[] = ['gaussian', 'gaussian', 'gaussian', 'gaussian'];

  it('produces symmetric weight matrix', () => {
    const result = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(result.weightMatrix[i]![j]).toBeCloseTo(result.weightMatrix[j]![i]!, 10);
      }
    }
  });

  it('produces non-negative weights', () => {
    const result = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(result.weightMatrix[i]![j]).toBeGreaterThanOrEqual(-1e-10);
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════
//  All-binary matches IsingFit-like behavior
// ═══════════════════════════════════════════════════════════

describe('fitMGM — all-binary', () => {
  const data = makeAllBinaryData(300, 4, 3, 42);
  const labels = ['A', 'B', 'C', 'D'];
  const nodeTypes: MgmNodeType[] = ['binary', 'binary', 'binary', 'binary'];

  it('produces symmetric weight matrix', () => {
    const result = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(result.weightMatrix[i]![j]).toBeCloseTo(result.weightMatrix[j]![i]!, 10);
      }
    }
  });

  it('detects chain structure (adjacent > non-adjacent)', () => {
    const result = fitMGM(data, labels, nodeTypes, { rule: 'OR' });
    const adj01 = result.weightMatrix[0]![1]!;
    const nonadj03 = result.weightMatrix[0]![3]!;
    expect(adj01).toBeGreaterThan(nonadj03);
  });
});

// ═══════════════════════════════════════════════════════════
//  Input validation
// ═══════════════════════════════════════════════════════════

describe('fitMGM — validation', () => {
  it('throws on non-binary data in binary column', () => {
    const data = [[1, 2], [0, 1]];
    expect(() => fitMGM(data, ['A', 'B'], ['binary', 'binary'])).toThrow(/non-binary/);
  });

  it('throws on negative count in poisson column', () => {
    const data = [[1, -1], [0, 2]];
    expect(() => fitMGM(data, ['A', 'B'], ['binary', 'poisson'])).toThrow(/non-negative integer/);
  });

  it('throws on float in poisson column', () => {
    const data = [[1, 1.5], [0, 2]];
    expect(() => fitMGM(data, ['A', 'B'], ['binary', 'poisson'])).toThrow(/non-negative integer/);
  });

  it('throws on empty data', () => {
    expect(() => fitMGM([], ['A'], ['gaussian'])).toThrow();
  });

  it('throws on mismatched nodeTypes length', () => {
    const data = [[1, 2], [3, 4]];
    expect(() => fitMGM(data, ['A', 'B'], ['gaussian'])).toThrow(/nodeTypes/);
  });
});

// ═══════════════════════════════════════════════════════════
//  Determinism
// ═══════════════════════════════════════════════════════════

describe('fitMGM — determinism', () => {
  it('same data produces same result', () => {
    const { data, labels, nodeTypes } = makeMixedData(200, 42);
    const r1 = fitMGM(data, labels, nodeTypes);
    const r2 = fitMGM(data, labels, nodeTypes);
    for (let i = 0; i < labels.length; i++) {
      for (let j = 0; j < labels.length; j++) {
        expect(r1.weightMatrix[i]![j]).toBe(r2.weightMatrix[i]![j]);
        expect(r1.signMatrix[i]![j]).toBe(r2.signMatrix[i]![j]);
      }
    }
  });
});
