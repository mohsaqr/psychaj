/**
 * Extended IsingFit R numerical equivalence tests
 * 6 additional datasets across varied structures and parameters.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { fitIsing } from '../../src/estimation/isingfit';

interface RResult { weightMatrix: number[][]; thresholds: number[] }
interface RDataset {
  description: string;
  data: Record<string, number[]>;
  n: number; p: number; labels: string[];
  and_rule?: RResult; or_rule?: RResult;
  gamma025?: RResult; gamma050?: RResult;
}

const gt: Record<string, RDataset> = JSON.parse(
  readFileSync(join(__dirname, '..', 'fixtures', 'ising-extended-ground-truth.json'), 'utf-8'),
);

function toRows(data: Record<string, number[]>, labels: string[]): number[][] {
  const n = data[labels[0]!]!.length;
  return Array.from({ length: n }, (_, i) => labels.map(l => data[l]![i]!));
}

function edges(m: number[][], thr = 1e-6): number {
  let c = 0;
  for (let i = 0; i < m.length; i++)
    for (let j = i + 1; j < m.length; j++)
      if (Math.abs(m[i]![j]!) > thr) c++;
  return c;
}

function structure(m: number[][], thr = 1e-6): boolean[][] {
  return m.map((row, i) => row.map((_, j) => Math.abs(m[i]![j]!) > thr));
}

function compareWeights(rM: number[][], jsM: number[][], p: number) {
  let maxDiff = 0, sumDiff = 0, count = 0, sharedEdges = 0;
  let structMismatch = 0;
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      const rw = Math.abs(rM[i]![j]!);
      const jsw = Math.abs(jsM[i]![j]!);
      const diff = Math.abs(rw - jsw);
      if (diff > maxDiff) maxDiff = diff;
      sumDiff += diff;
      count++;
      if (rw > 1e-6 && jsw > 1e-6) sharedEdges++;
      if ((rw > 1e-6) !== (jsw > 1e-6)) structMismatch++;
    }
  }
  return { maxDiff, meanDiff: sumDiff / count, sharedEdges, structMismatch, totalPairs: count };
}

function compareThresholds(rT: number[], jsT: number[]) {
  let maxDiff = 0, sumDiff = 0;
  for (let i = 0; i < rT.length; i++) {
    const diff = Math.abs(rT[i]! - jsT[i]!);
    if (diff > maxDiff) maxDiff = diff;
    sumDiff += diff;
  }
  return { maxDiff, meanDiff: sumDiff / rT.length };
}

// ═══════════════════════════════════════════════════════════

describe('IsingFit extended — Dataset 4 (dense, p=4, n=500)', () => {
  const ds = gt.dataset4_dense!;
  const data = toRows(ds.data, ds.labels);

  it('AND rule: structure and weight match', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareWeights(ds.and_rule!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBe(0);
    expect(cmp.maxDiff).toBeLessThan(0.02);
  });

  it('OR rule: structure and weight match', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'OR' });
    const cmp = compareWeights(ds.or_rule!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBe(0);
    expect(cmp.maxDiff).toBeLessThan(0.02);
  });

  it('thresholds close to R', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareThresholds(ds.and_rule!.thresholds, result.thresholds);
    expect(cmp.maxDiff).toBeLessThan(0.05);
  });
});

describe('IsingFit extended — Dataset 5 (large chain, p=8, n=500)', () => {
  const ds = gt.dataset5_large_chain!;
  const data = toRows(ds.data, ds.labels);

  it('structure and weight match', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareWeights(ds.and_rule!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBe(0);
    expect(cmp.maxDiff).toBeLessThan(0.01);
  });

  it('thresholds close to R', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareThresholds(ds.and_rule!.thresholds, result.thresholds);
    expect(cmp.maxDiff).toBeLessThan(0.01);
  });
});

describe('IsingFit extended — Dataset 6 (two clusters, p=6, n=400)', () => {
  const ds = gt.dataset6_two_clusters!;
  const data = toRows(ds.data, ds.labels);

  it('structure and weight match', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareWeights(ds.and_rule!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBe(0);
    expect(cmp.maxDiff).toBeLessThan(0.01);
  });
});

describe('IsingFit extended — Dataset 7 (unbalanced prevalence, p=5, n=300)', () => {
  const ds = gt.dataset7_unbalanced!;
  const data = toRows(ds.data, ds.labels);

  it('structure and weight match', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareWeights(ds.and_rule!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBe(0);
    expect(cmp.maxDiff).toBeLessThan(0.01);
  });

  it('thresholds close to R', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareThresholds(ds.and_rule!.thresholds, result.thresholds);
    expect(cmp.maxDiff).toBeLessThan(0.01);
  });
});

describe('IsingFit extended — Dataset 8 (large n=1000, p=5)', () => {
  const ds = gt.dataset8_large_n!;
  const data = toRows(ds.data, ds.labels);

  it('structure and weight match', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareWeights(ds.and_rule!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBe(0);
    expect(cmp.maxDiff).toBeLessThan(0.01);
  });

  it('thresholds close to R', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareThresholds(ds.and_rule!.thresholds, result.thresholds);
    expect(cmp.maxDiff).toBeLessThan(0.01);
  });
});

describe('IsingFit extended — Dataset 9 (gamma comparison, p=5, n=400)', () => {
  const ds = gt.dataset9_gamma_comparison!;
  const data = toRows(ds.data, ds.labels);

  it('gamma=0.25 structure matches R', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const cmp = compareWeights(ds.gamma025!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBeLessThanOrEqual(1);
  });

  it('gamma=0.5 structure matches R', () => {
    const result = fitIsing(data, ds.labels, { gamma: 0.5, rule: 'AND' });
    const cmp = compareWeights(ds.gamma050!.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBeLessThanOrEqual(1);
  });

  it('higher gamma produces sparser or equal network', () => {
    const r025 = fitIsing(data, ds.labels, { gamma: 0.25, rule: 'AND' });
    const r050 = fitIsing(data, ds.labels, { gamma: 0.5, rule: 'AND' });
    expect(edges(r050.weightMatrix)).toBeLessThanOrEqual(edges(r025.weightMatrix));
  });
});

// ═══════════════════════════════════════════════════════════
//  Grand summary
// ═══════════════════════════════════════════════════════════

describe('IsingFit extended — grand summary', () => {
  it('prints all comparisons', () => {
    const allDatasets = [
      { key: 'dataset4_dense', gamma: 0.25, ruleKey: 'and_rule' },
      { key: 'dataset5_large_chain', gamma: 0.25, ruleKey: 'and_rule' },
      { key: 'dataset6_two_clusters', gamma: 0.25, ruleKey: 'and_rule' },
      { key: 'dataset7_unbalanced', gamma: 0.25, ruleKey: 'and_rule' },
      { key: 'dataset8_large_n', gamma: 0.25, ruleKey: 'and_rule' },
      { key: 'dataset9_gamma_comparison', gamma: 0.25, ruleKey: 'gamma025' },
      { key: 'dataset9_gamma_comparison', gamma: 0.5, ruleKey: 'gamma050' },
    ] as const;

    let totalPairs = 0, totalStructMismatch = 0, globalMaxDiff = 0;
    let totalThresholdChecks = 0, globalMaxThreshDiff = 0;

    for (const { key, gamma, ruleKey } of allDatasets) {
      const ds = gt[key]!;
      const rResult = (ds as any)[ruleKey] as RResult;
      const data = toRows(ds.data, ds.labels);
      const result = fitIsing(data, ds.labels, { gamma, rule: 'AND' });

      const wCmp = compareWeights(rResult.weightMatrix, result.weightMatrix, ds.p);
      const tCmp = compareThresholds(rResult.thresholds, result.thresholds);

      totalPairs += wCmp.totalPairs;
      totalStructMismatch += wCmp.structMismatch;
      if (wCmp.maxDiff > globalMaxDiff) globalMaxDiff = wCmp.maxDiff;
      totalThresholdChecks += rResult.thresholds.length;
      if (tCmp.maxDiff > globalMaxThreshDiff) globalMaxThreshDiff = tCmp.maxDiff;

      console.log(`  ${key}(γ=${gamma}): R=${edges(rResult.weightMatrix)}edges JS=${edges(result.weightMatrix)}edges structΔ=${wCmp.structMismatch} maxWΔ=${wCmp.maxDiff.toFixed(4)} meanWΔ=${wCmp.meanDiff.toFixed(4)} maxTΔ=${tCmp.maxDiff.toFixed(4)}`);
    }

    console.log(`\n  TOTAL: ${totalPairs} edge pairs, ${totalStructMismatch} structure mismatches, globalMaxWeightDiff=${globalMaxDiff.toFixed(4)}, ${totalThresholdChecks} threshold checks, globalMaxThreshDiff=${globalMaxThreshDiff.toFixed(4)}`);
  });
});
