import { describe, it, expect } from 'vitest';
import { SeededRNG } from '../../src/core/rng';

describe('SeededRNG', () => {
  it('produces deterministic sequences', () => {
    const rng1 = new SeededRNG(42);
    const rng2 = new SeededRNG(42);
    const seq1 = Array.from({ length: 10 }, () => rng1.random());
    const seq2 = Array.from({ length: 10 }, () => rng2.random());
    expect(seq1).toEqual(seq2);
  });

  it('different seeds produce different sequences', () => {
    const rng1 = new SeededRNG(42);
    const rng2 = new SeededRNG(123);
    const seq1 = Array.from({ length: 10 }, () => rng1.random());
    const seq2 = Array.from({ length: 10 }, () => rng2.random());
    expect(seq1).not.toEqual(seq2);
  });

  it('random() values are in [0, 1)', () => {
    const rng = new SeededRNG(42);
    for (let i = 0; i < 1000; i++) {
      const v = rng.random();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('randInt(n) values are in [0, n)', () => {
    const rng = new SeededRNG(42);
    for (let i = 0; i < 100; i++) {
      const v = rng.randInt(10);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(10);
      expect(Number.isInteger(v)).toBe(true);
    }
  });

  it('permutation returns all indices', () => {
    const rng = new SeededRNG(42);
    const perm = rng.permutation(10);
    expect(perm.length).toBe(10);
    expect([...perm].sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('choiceWithoutReplacement returns unique elements', () => {
    const rng = new SeededRNG(42);
    const result = rng.choiceWithoutReplacement(20, 10);
    expect(result.length).toBe(10);
    expect(new Set(result).size).toBe(10);
    for (const v of result) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(20);
    }
  });

  it('shuffle preserves elements', () => {
    const rng = new SeededRNG(42);
    const arr = [1, 2, 3, 4, 5];
    rng.shuffle(arr);
    expect([...arr].sort()).toEqual([1, 2, 3, 4, 5]);
  });
});
