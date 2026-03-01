import { describe, it, expect } from 'vitest';
import {
  lgamma, gammaP, chiSqCDF, betaI, fDistCDF, tDistCDF, tDistInv,
  normalCDF, percentile, pearsonCorr,
} from '../../src/core/stats';

describe('lgamma', () => {
  it('known values', () => {
    expect(lgamma(1)).toBeCloseTo(0, 10);
    expect(lgamma(2)).toBeCloseTo(0, 10);
    expect(lgamma(5)).toBeCloseTo(Math.log(24), 8);
    expect(lgamma(0.5)).toBeCloseTo(Math.log(Math.sqrt(Math.PI)), 8);
  });
});

describe('chiSqCDF', () => {
  it('df=1, x=3.84 ≈ 0.95', () => {
    expect(chiSqCDF(3.84, 1)).toBeCloseTo(0.95, 2);
  });
  it('returns 0 for x <= 0', () => {
    expect(chiSqCDF(0, 5)).toBe(0);
    expect(chiSqCDF(-1, 5)).toBe(0);
  });
});

describe('tDistCDF', () => {
  it('t=0 gives 0.5', () => {
    expect(tDistCDF(0, 10)).toBeCloseTo(0.5, 10);
  });
  it('t=1.96, df=Inf ≈ 0.975 (normal)', () => {
    expect(tDistCDF(1.96, 100000)).toBeCloseTo(0.975, 2);
  });
});

describe('tDistInv', () => {
  it('p=0.5 gives 0', () => {
    expect(tDistInv(0.5, 10)).toBe(0);
  });
  it('round-trip: tDistCDF(tDistInv(p, df), df) ≈ p', () => {
    const p = 0.975;
    const df = 20;
    const t = tDistInv(p, df);
    expect(tDistCDF(t, df)).toBeCloseTo(p, 5);
  });
});

describe('normalCDF', () => {
  it('z=0 gives 0.5', () => {
    expect(normalCDF(0)).toBeCloseTo(0.5, 8);
  });
  it('z=1.96 ≈ 0.975', () => {
    expect(normalCDF(1.96)).toBeCloseTo(0.975, 2);
  });
});

describe('percentile', () => {
  it('median of [1,2,3,4,5]', () => {
    expect(percentile(new Float64Array([1, 2, 3, 4, 5]), 0.5)).toBe(3);
  });
  it('0th percentile = first element', () => {
    expect(percentile(new Float64Array([10, 20, 30]), 0)).toBe(10);
  });
  it('100th percentile = last element', () => {
    expect(percentile(new Float64Array([10, 20, 30]), 1)).toBe(30);
  });
});

describe('pearsonCorr', () => {
  it('perfect positive correlation', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([2, 4, 6, 8, 10]);
    expect(pearsonCorr(a, b)).toBeCloseTo(1, 10);
  });
  it('perfect negative correlation', () => {
    const a = new Float64Array([1, 2, 3, 4, 5]);
    const b = new Float64Array([10, 8, 6, 4, 2]);
    expect(pearsonCorr(a, b)).toBeCloseTo(-1, 10);
  });
  it('zero correlation', () => {
    const a = new Float64Array([1, -1, 1, -1]);
    const b = new Float64Array([1, 1, -1, -1]);
    expect(pearsonCorr(a, b)).toBeCloseTo(0, 10);
  });
});
