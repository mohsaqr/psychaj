import { describe, it, expect } from 'vitest';
import { fitIsing } from '../../src/estimation/isingfit';

// ── Deterministic binary data generator ──
function makeBinaryChainData(n: number, p: number, strength: number, seed: number): number[][] {
  // Generate data from a chain graph: 1-2-3-...-p
  // Each node depends on its neighbor with given strength
  let s = seed;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };

  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(p);
    // First node: 50/50
    row[0] = next() < 0.5 ? 1 : 0;
    for (let j = 1; j < p; j++) {
      // P(x_j = 1 | x_{j-1}) depends on neighbor
      const eta = strength * (row[j - 1]! - 0.5);
      const prob = 1 / (1 + Math.exp(-eta));
      row[j] = next() < prob ? 1 : 0;
    }
    data.push(row);
  }
  return data;
}

function makeIndependentBinaryData(n: number, p: number, seed: number): number[][] {
  let s = seed;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  return Array.from({ length: n }, () =>
    Array.from({ length: p }, () => next() < 0.5 ? 1 : 0),
  );
}

const labels4 = ['A', 'B', 'C', 'D'];
const labels5 = ['A', 'B', 'C', 'D', 'E'];

// ═══════════════════════════════════════════════════════════
//  Structure
// ═══════════════════════════════════════════════════════════

describe('fitIsing — structure', () => {
  const data = makeBinaryChainData(500, 4, 3, 42);

  it('returns p×p symmetric weight matrix', () => {
    const result = fitIsing(data, labels4);
    expect(result.weightMatrix.length).toBe(4);
    for (let i = 0; i < 4; i++) {
      expect(result.weightMatrix[i]!.length).toBe(4);
      for (let j = 0; j < 4; j++) {
        expect(result.weightMatrix[i]![j]).toBeCloseTo(result.weightMatrix[j]![i]!, 10);
      }
    }
  });

  it('has zero diagonal', () => {
    const result = fitIsing(data, labels4);
    for (let i = 0; i < 4; i++) {
      expect(result.weightMatrix[i]![i]).toBe(0);
    }
  });

  it('returns correct metadata', () => {
    const result = fitIsing(data, labels4, { gamma: 0.5, rule: 'OR' });
    expect(result.labels).toEqual(labels4);
    expect(result.gamma).toBe(0.5);
    expect(result.rule).toBe('OR');
    expect(result.nObs).toBe(500);
    expect(result.thresholds.length).toBe(4);
    expect(result.lambdas.length).toBe(4);
  });
});

// ═══════════════════════════════════════════════════════════
//  Chain structure recovery
// ═══════════════════════════════════════════════════════════

describe('fitIsing — chain recovery', () => {
  // Strong chain: 1-2-3-4 with strength=4
  const data = makeBinaryChainData(500, 4, 4, 42);

  it('detects adjacent edges (chain neighbors)', () => {
    const result = fitIsing(data, labels4);
    // Adjacent pairs should have nonzero weights
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(result.weightMatrix[i]![i + 1]!)).toBeGreaterThan(0);
    }
  });

  it('non-adjacent edges are weaker than adjacent', () => {
    const result = fitIsing(data, labels4, { rule: 'OR' });
    // 0-1 edge should be stronger than 0-2 or 0-3
    const adj01 = Math.abs(result.weightMatrix[0]![1]!);
    const nonadj02 = Math.abs(result.weightMatrix[0]![2]!);
    const nonadj03 = Math.abs(result.weightMatrix[0]![3]!);
    expect(adj01).toBeGreaterThan(nonadj02);
    expect(adj01).toBeGreaterThan(nonadj03);
  });
});

// ═══════════════════════════════════════════════════════════
//  AND vs OR rule
// ═══════════════════════════════════════════════════════════

describe('fitIsing — AND vs OR', () => {
  const data = makeBinaryChainData(300, 5, 3, 42);

  it('AND is sparser than or equal to OR', () => {
    const andResult = fitIsing(data, labels5, { rule: 'AND' });
    const orResult = fitIsing(data, labels5, { rule: 'OR' });

    let andEdges = 0;
    let orEdges = 0;
    for (let i = 0; i < 5; i++) {
      for (let j = i + 1; j < 5; j++) {
        if (Math.abs(andResult.weightMatrix[i]![j]!) > 1e-10) andEdges++;
        if (Math.abs(orResult.weightMatrix[i]![j]!) > 1e-10) orEdges++;
      }
    }
    expect(andEdges).toBeLessThanOrEqual(orEdges);
  });
});

// ═══════════════════════════════════════════════════════════
//  Independent data → sparse network
// ═══════════════════════════════════════════════════════════

describe('fitIsing — independent data', () => {
  const data = makeIndependentBinaryData(300, 4, 42);

  it('produces sparse network for independent data', () => {
    const result = fitIsing(data, labels4);
    let edgeCount = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        if (Math.abs(result.weightMatrix[i]![j]!) > 1e-10) edgeCount++;
      }
    }
    // Independent data should have few or no edges
    expect(edgeCount).toBeLessThanOrEqual(3);
  });
});

// ═══════════════════════════════════════════════════════════
//  Input validation
// ═══════════════════════════════════════════════════════════

describe('fitIsing — validation', () => {
  it('throws on non-binary data', () => {
    const data = [[0, 1], [1, 2], [0, 0]];
    expect(() => fitIsing(data, ['A', 'B'])).toThrow(/non-binary/);
  });

  it('throws on empty data', () => {
    expect(() => fitIsing([], ['A', 'B'])).toThrow();
  });

  it('throws on single variable', () => {
    expect(() => fitIsing([[0], [1]], ['A'])).toThrow(/at least 2/);
  });

  it('throws on zero-variance column', () => {
    const data = [[0, 1], [0, 0], [0, 1]];
    expect(() => fitIsing(data, ['A', 'B'])).toThrow(/zero variance/);
  });

  it('throws on mismatched dimensions', () => {
    const data = [[0, 1, 0], [1, 0, 1]];
    expect(() => fitIsing(data, ['A', 'B'])).toThrow();
  });
});

// ═══════════════════════════════════════════════════════════
//  Determinism
// ═══════════════════════════════════════════════════════════

describe('fitIsing — determinism', () => {
  it('same data produces same result', () => {
    const data = makeBinaryChainData(200, 4, 3, 42);
    const r1 = fitIsing(data, labels4);
    const r2 = fitIsing(data, labels4);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(r1.weightMatrix[i]![j]).toBe(r2.weightMatrix[i]![j]);
      }
    }
  });
});
