/**
 * Deterministic 32-bit PRNG (Mulberry32) matching Python RngState bit-for-bit.
 */

export function hashSeed(seed) {
  if (typeof seed === 'number' && Number.isFinite(seed)) return seed >>> 0;
  const str = String(seed ?? 'gwent');
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export class RngState {
  constructor(seed = 1) {
    this.state = hashSeed(seed);
  }

  clone() {
    const r = new RngState(0);
    r.state = this.state;
    return r;
  }

  nextFloat() {
    let t = (this.state = (this.state + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ (t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0))) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  }

  randInt(n) {
    if (n <= 0) return 0;
    return Math.floor(this.nextFloat() * n);
  }

  shuffle(arr) {
    const out = arr.slice();
    for (let i = out.length - 1; i > 0; i--) {
      const j = this.randInt(i + 1);
      const tmp = out[i];
      out[i] = out[j];
      out[j] = tmp;
    }
    return out;
  }

  pickRandom(arr) {
    if (!arr || !arr.length) return null;
    return arr[this.randInt(arr.length)];
  }
}
