/**
 * Extended MGM R numerical equivalence tests
 * 6 additional datasets covering all-gaussian, all-binary, all-poisson, and mixed types.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { fitMGM } from '../../src/estimation/mgm';
import type { MgmNodeType } from '../../src/core/types';

interface RDataset {
  description: string;
  data: Record<string, number[]>;
  n: number; p: number; labels: string[];
  types: string[]; levels: number[];
  weightMatrix: number[][];
  signMatrix: number[][];
}

const gt: Record<string, RDataset> = JSON.parse(
  readFileSync(join(__dirname, '..', 'fixtures', 'mgm-extended-ground-truth.json'), 'utf-8'),
);

function toRows(data: Record<string, number[]>, labels: string[]): number[][] {
  const n = data[labels[0]!]!.length;
  return Array.from({ length: n }, (_, i) => labels.map(l => data[l]![i]!));
}

function rType(t: string): MgmNodeType {
  if (t === 'g') return 'gaussian';
  if (t === 'c') return 'binary';
  if (t === 'p') return 'poisson';
  throw new Error(`Unknown type: ${t}`);
}

function edges(m: number[][], thr = 1e-6): number {
  let c = 0;
  for (let i = 0; i < m.length; i++)
    for (let j = i + 1; j < m.length; j++)
      if (Math.abs(m[i]![j]!) > thr) c++;
  return c;
}

function compare(rM: number[][], jsM: number[][], p: number) {
  let maxDiff = 0, sumDiff = 0, count = 0, sharedEdges = 0, structMismatch = 0;
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

function signAgreement(rS: number[][], jsS: number[][], rW: number[][], jsW: number[][], p: number) {
  let agree = 0, total = 0;
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      // Only compare signs where both have edges
      if (Math.abs(rW[i]![j]!) > 1e-6 && Math.abs(jsW[i]![j]!) > 1e-6) {
        const rSign = rS[i]![j]!;
        const jsSign = jsS[i]![j]!;
        // R uses NA (null in JSON) for some signs — skip those
        if (rSign !== null && rSign !== undefined && !isNaN(rSign)) {
          total++;
          if (rSign === jsSign) agree++;
        }
      }
    }
  }
  return { agree, total };
}

// ═══════════════════════════════════════════════════════════

describe('MGM extended — Dataset 4 (gaussian chain, p=5, n=500)', () => {
  const ds = gt.dataset4_gaussian_chain!;
  const data = toRows(ds.data, ds.labels);
  const types = ds.types.map(rType);

  it('structure matches R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBeLessThanOrEqual(1);
  });

  it('weights close to R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.maxDiff).toBeLessThan(0.1);
  });

  it('sign agreement', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const sa = signAgreement(ds.signMatrix, result.signMatrix, ds.weightMatrix, result.weightMatrix, ds.p);
    if (sa.total > 0) expect(sa.agree / sa.total).toBeGreaterThanOrEqual(0.8);
  });
});

describe('MGM extended — Dataset 5 (binary chain, p=6, n=500)', () => {
  const ds = gt.dataset5_binary_chain6!;
  const data = toRows(ds.data, ds.labels);
  const types = ds.types.map(rType);

  it('structure matches R (at most 2 disagreements)', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBeLessThanOrEqual(2);
  });

  it('weights close to R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.maxDiff).toBeLessThan(0.15);
  });
});

describe('MGM extended — Dataset 6 (mixed 6-node, 2g+2c+2p, n=600)', () => {
  const ds = gt.dataset6_mixed_6node!;
  const data = toRows(ds.data, ds.labels);
  const types = ds.types.map(rType);

  it('structure within ±3 edges', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const rE = edges(ds.weightMatrix);
    const jsE = edges(result.weightMatrix);
    expect(Math.abs(rE - jsE)).toBeLessThanOrEqual(3);
  });

  it('shared edges weights reasonable', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    if (cmp.sharedEdges > 0) {
      expect(cmp.maxDiff).toBeLessThan(0.3);
    }
  });
});

describe('MGM extended — Dataset 7 (gaussian varied, p=4, n=400)', () => {
  const ds = gt.dataset7_gaussian_varied!;
  const data = toRows(ds.data, ds.labels);
  const types = ds.types.map(rType);

  it('structure matches R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBeLessThanOrEqual(1);
  });

  it('weights close to R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.maxDiff).toBeLessThan(0.1);
  });

  it('signs match R on shared edges', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const sa = signAgreement(ds.signMatrix, result.signMatrix, ds.weightMatrix, result.weightMatrix, ds.p);
    if (sa.total > 0) expect(sa.agree / sa.total).toBeGreaterThanOrEqual(0.8);
  });
});

describe('MGM extended — Dataset 8 (large-n mixed, p=4, n=1000)', () => {
  const ds = gt.dataset8_large_n_mixed!;
  const data = toRows(ds.data, ds.labels);
  const types = ds.types.map(rType);

  it('structure within ±2 edges', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const rE = edges(ds.weightMatrix);
    const jsE = edges(result.weightMatrix);
    expect(Math.abs(rE - jsE)).toBeLessThanOrEqual(2);
  });

  it('weights close to R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.maxDiff).toBeLessThan(0.2);
  });
});

describe('MGM extended — Dataset 9 (all-poisson, p=4, n=400)', () => {
  const ds = gt.dataset9_all_poisson!;
  const data = toRows(ds.data, ds.labels);
  const types = ds.types.map(rType);

  it('structure matches R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.structMismatch).toBeLessThanOrEqual(1);
  });

  it('weights close to R', () => {
    const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });
    const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
    expect(cmp.maxDiff).toBeLessThan(0.15);
  });
});

// ═══════════════════════════════════════════════════════════
//  Grand summary
// ═══════════════════════════════════════════════════════════

describe('MGM extended — grand summary', () => {
  it('prints all comparisons', () => {
    let totalPairs = 0, totalStructMismatch = 0, globalMaxDiff = 0;
    let totalSignChecks = 0, totalSignAgree = 0;

    for (const key of Object.keys(gt)) {
      const ds = gt[key]!;
      const data = toRows(ds.data, ds.labels);
      const types = ds.types.map(rType);
      const result = fitMGM(data, ds.labels, types, { gamma: 0.25, rule: 'AND' });

      const cmp = compare(ds.weightMatrix, result.weightMatrix, ds.p);
      const sa = signAgreement(ds.signMatrix, result.signMatrix, ds.weightMatrix, result.weightMatrix, ds.p);

      totalPairs += cmp.totalPairs;
      totalStructMismatch += cmp.structMismatch;
      if (cmp.maxDiff > globalMaxDiff) globalMaxDiff = cmp.maxDiff;
      totalSignChecks += sa.total;
      totalSignAgree += sa.agree;

      console.log(`  ${key}: R=${edges(ds.weightMatrix)}edges JS=${edges(result.weightMatrix)}edges structΔ=${cmp.structMismatch} maxWΔ=${cmp.maxDiff.toFixed(4)} meanWΔ=${cmp.meanDiff.toFixed(4)} signs=${sa.agree}/${sa.total}`);
    }

    console.log(`\n  TOTAL: ${totalPairs} edge pairs, ${totalStructMismatch} structure mismatches, globalMaxWeightDiff=${globalMaxDiff.toFixed(4)}, signAgreement=${totalSignAgree}/${totalSignChecks}`);
  });
});
