import { describe, it, expect } from 'vitest';
import { fitMlVAR, computeImpulseResponse } from '../../src/models/mlvar';

function makeEmaData(opts: {
  nSubjects: number;
  nDays: number;
  nBeeps: number;
  vars: string[];
  seed: number;
}): Record<string, string | number>[] {
  const { nSubjects, nDays, nBeeps, vars, seed } = opts;
  let s = seed;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s / 0x100000000) * 6 + 1; };
  const rows: Record<string, string | number>[] = [];
  for (let id = 1; id <= nSubjects; id++) {
    for (let day = 1; day <= nDays; day++) {
      for (let beep = 1; beep <= nBeeps; beep++) {
        const row: Record<string, string | number> = {
          id: `S${id}`,
          day,
          beep,
        };
        for (const v of vars) row[v] = Math.round(next());
        rows.push(row);
      }
    }
  }
  return rows;
}

describe('fitMlVAR', () => {
  it('produces d×d temporal, contemporaneous, and between matrices', () => {
    const vars = ['a', 'b', 'c'];
    const rows = makeEmaData({ nSubjects: 5, nDays: 5, nBeeps: 8, vars, seed: 42 });
    const result = fitMlVAR(rows, {
      vars,
      idvar: 'id',
      dayvar: 'day',
      beepvar: 'beep',
    });
    expect(result.temporal.length).toBe(3);
    expect(result.contemporaneous.length).toBe(3);
    expect(result.between.length).toBe(3);
    expect(result.labels).toEqual(vars);
    expect(result.nSubjects).toBe(5);
  });

  it('coefs table has d² entries', () => {
    const vars = ['x', 'y'];
    const rows = makeEmaData({ nSubjects: 10, nDays: 3, nBeeps: 5, vars, seed: 99 });
    const result = fitMlVAR(rows, {
      vars,
      idvar: 'id',
      dayvar: 'day',
      beepvar: 'beep',
    });
    expect(result.coefs.length).toBe(4); // 2×2
    for (const c of result.coefs) {
      expect(typeof c.estimate).toBe('number');
      expect(typeof c.pValue).toBe('number');
      expect(c.ci.length).toBe(2);
    }
  });

  it('throws on insufficient data', () => {
    const vars = ['a', 'b', 'c'];
    expect(() =>
      fitMlVAR([{ id: 'S1', day: 1, beep: 1, a: 1, b: 2, c: 3 }], {
        vars,
        idvar: 'id',
        dayvar: 'day',
        beepvar: 'beep',
      }),
    ).toThrow();
  });

  it('throws on fewer than 2 variables', () => {
    expect(() =>
      fitMlVAR([], { vars: ['a'], idvar: 'id' }),
    ).toThrow('at least 2');
  });
});

describe('computeImpulseResponse', () => {
  it('returns d trajectories over nSteps', () => {
    const temporal = [[0.3, 0.1], [0.2, 0.5]];
    const labels = ['x', 'y'];
    const ir = computeImpulseResponse(temporal, labels, 0, 10);
    expect(ir.trajectories.length).toBe(2);
    expect(ir.trajectories[0]!.length).toBe(10);
    expect(ir.shockedVar).toBe('x');
    // First time step: only shocked variable
    expect(ir.trajectories[0]![0]).toBe(1);
    expect(ir.trajectories[1]![0]).toBe(0);
  });

  it('spectral radius < 1 means shocks decay', () => {
    const temporal = [[0.1, 0.05], [0.05, 0.1]];
    const labels = ['a', 'b'];
    const ir = computeImpulseResponse(temporal, labels, 0, 50);
    expect(ir.spectralRadius).toBeLessThan(1);
    // Last step values should be near zero
    expect(Math.abs(ir.trajectories[0]![49]!)).toBeLessThan(0.01);
  });
});
