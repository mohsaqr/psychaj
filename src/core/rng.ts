/**
 * Seeded pseudo-random number generator using xoshiro128** algorithm.
 * Own implementation — no tnaj dependency.
 * Produces identical sequences to tnaj's SeededRNG for reproducibility.
 */

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

function splitmix32(seed: number): number {
  seed = (seed + 0x9E3779B9) >>> 0;
  seed = Math.imul(seed ^ (seed >>> 16), 0x85EBCA6B) >>> 0;
  seed = Math.imul(seed ^ (seed >>> 13), 0xC2B2AE35) >>> 0;
  return (seed ^ (seed >>> 16)) >>> 0;
}

export class SeededRNG {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;

  constructor(seed: number) {
    seed = seed >>> 0;
    this.s0 = splitmix32(seed);
    this.s1 = splitmix32(this.s0);
    this.s2 = splitmix32(this.s1);
    this.s3 = splitmix32(this.s2);
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) {
      this.s0 = 1;
    }
  }

  /** Generate next random 32-bit unsigned integer (xoshiro128**). */
  next(): number {
    const result = (Math.imul(rotl(Math.imul(this.s1, 5), 7), 9)) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 = (this.s2 ^ this.s0) >>> 0;
    this.s3 = (this.s3 ^ this.s1) >>> 0;
    this.s1 = (this.s1 ^ this.s2) >>> 0;
    this.s0 = (this.s0 ^ this.s3) >>> 0;
    this.s2 = (this.s2 ^ t) >>> 0;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Generate a random float in [0, 1). */
  random(): number {
    return this.next() / 0x100000000;
  }

  /** Generate a random integer in [0, max). */
  randInt(max: number): number {
    return Math.floor(this.random() * max);
  }

  /** Fisher-Yates shuffle (in-place). */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randInt(i + 1);
      const tmp = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = tmp;
    }
    return arr;
  }

  /** Generate a random permutation of indices [0, n). */
  permutation(n: number): number[] {
    const arr = Array.from({ length: n }, (_, i) => i);
    return this.shuffle(arr);
  }

  /** Random choice with replacement: pick `size` items from [0, n). */
  choice(n: number, size: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < size; i++) {
      result.push(this.randInt(n));
    }
    return result;
  }

  /** Random choice WITHOUT replacement: pick `size` items from [0, n). */
  choiceWithoutReplacement(n: number, size: number): number[] {
    if (size > n) throw new Error(`Cannot choose ${size} from ${n} without replacement`);
    const pool = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < size; i++) {
      const j = i + this.randInt(n - i);
      const tmp = pool[i]!;
      pool[i] = pool[j]!;
      pool[j] = tmp;
    }
    return pool.slice(0, size);
  }
}
