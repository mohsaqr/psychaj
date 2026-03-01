import { describe, it, expect } from 'vitest';
import { fitGraphicalVAR } from '../../src/models/graphicalvar';
import { computePDC } from '../../src/models/pdc';
import { computePCC } from '../../src/models/pcc';

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
        const row: Record<string, string | number> = { id: `S${id}`, day, beep };
        for (const v of vars) row[v] = Math.round(next());
        rows.push(row);
      }
    }
  }
  return rows;
}

describe('fitGraphicalVAR', () => {
  it('produces result with all required fields', () => {
    const vars = ['a', 'b', 'c'];
    const rows = makeEmaData({ nSubjects: 5, nDays: 5, nBeeps: 8, vars, seed: 42 });
    const result = fitGraphicalVAR(rows, {
      vars,
      idvar: 'id',
      dayvar: 'day',
      beepvar: 'beep',
      nLambda: 5,
    });
    expect(result.temporal.length).toBe(3);
    expect(result.PDC.length).toBe(3);
    expect(result.PCC.length).toBe(3);
    expect(result.kappa.length).toBe(3);
    expect(result.beta.length).toBe(4); // d+1 (with intercept)
    expect(result.labels).toEqual(vars);
    expect(result.nSubjects).toBe(5);
    expect(result.lambda_beta).toBeGreaterThan(0);
    expect(result.lambda_kappa).toBeGreaterThan(0);
  });

  it('throws on fewer than 2 variables', () => {
    expect(() =>
      fitGraphicalVAR([], { vars: ['a'], idvar: 'id' }),
    ).toThrow('at least 2');
  });

  it('deterministic with same data', () => {
    const vars = ['x', 'y'];
    const rows = makeEmaData({ nSubjects: 3, nDays: 5, nBeeps: 5, vars, seed: 77 });
    const r1 = fitGraphicalVAR(rows, { vars, idvar: 'id', dayvar: 'day', beepvar: 'beep', nLambda: 3 });
    const r2 = fitGraphicalVAR(rows, { vars, idvar: 'id', dayvar: 'day', beepvar: 'beep', nLambda: 3 });
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(r1.temporal[i]![j]).toBe(r2.temporal[i]![j]);
      }
    }
  });
});

describe('computePDC', () => {
  it('zero beta gives zero PDC', () => {
    const beta = [[0, 0], [0, 0]];
    const kappa = [[1, 0], [0, 1]];
    const pdc = computePDC(beta, kappa);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(pdc[i]![j]).toBe(0);
      }
    }
  });

  it('PDC values bounded in [-1, 1]', () => {
    const beta = [[0.3, -0.2], [0.1, 0.5]];
    const kappa = [[2, -0.5], [-0.5, 3]];
    const pdc = computePDC(beta, kappa);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(Math.abs(pdc[i]![j]!)).toBeLessThanOrEqual(1 + 1e-10);
      }
    }
  });
});

describe('computePCC', () => {
  it('PCC values bounded in [-1, 1]', () => {
    const kappa = [[2, -0.5, 0.1], [-0.5, 3, -0.3], [0.1, -0.3, 1.5]];
    const pcc = computePCC(kappa);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(Math.abs(pcc[i]![j]!)).toBeLessThanOrEqual(1 + 1e-10);
      }
    }
  });
});
