import { describe, expect, it } from 'vitest';
import { mergeHistory } from './history.mjs';

describe('mergeHistory', () => {
  it('rejects a non-array import with the same message both pages used to raise inline', () => {
    expect(() => mergeHistory({ not: 'an array' }, [])).toThrow('not an array');
    expect(() => mergeHistory('nope', [])).toThrow('not an array');
    expect(() => mergeHistory(null, [])).toThrow('not an array');
  });

  it('an empty import returns the existing history unchanged in order', () => {
    const existing = [{ time: 300 }, { time: 200 }, { time: 100 }];
    expect(mergeHistory([], existing)).toEqual(existing);
  });

  it('sorts the merged result newest-first, regardless of input order', () => {
    // The bug this module exists to fix: naive concat kept each array's own
    // internal order but did not interleave them by time. An import spanning
    // several sessions, merged with an existing history, must come out sorted
    // as one timeline — not "every imported entry, then every existing one."
    const incoming = [{ time: 500, tag: 'imported' }, { time: 150, tag: 'imported' }];
    const existing = [{ time: 400, tag: 'existing' }, { time: 300, tag: 'existing' }];

    const merged = mergeHistory(incoming, existing);

    expect(merged.map((e) => e.time)).toEqual([500, 400, 300, 150]);
  });

  it('is still correct when every imported entry predates every existing one', () => {
    // The one case naive concat happened to get right — must not regress it.
    const incoming = [{ time: 20 }, { time: 10 }];
    const existing = [{ time: 40 }, { time: 30 }];
    expect(mergeHistory(incoming, existing).map((e) => e.time)).toEqual([40, 30, 20, 10]);
  });

  it('treats a missing time as the epoch, sorting it to the end rather than corrupting the sort', () => {
    const incoming = [{ tag: 'no-time' }, { time: 200 }];
    const merged = mergeHistory(incoming, [{ time: 100 }]);
    expect(merged.map((e) => e.time || 'MISSING')).toEqual([200, 100, 'MISSING']);
  });

  it('does not mutate either input array', () => {
    const incoming = [{ time: 2 }];
    const existing = [{ time: 1 }];
    mergeHistory(incoming, existing);
    expect(incoming).toEqual([{ time: 2 }]);
    expect(existing).toEqual([{ time: 1 }]);
  });
});
