import { describe, expect, it } from 'vitest';
import {
  CHART_HEIGHT,
  CHART_PAD,
  CHART_WIDTH,
  LIFT_COLORS,
  aggregate,
  dayKey,
  dayShort,
  linePath,
  liftOf,
  xPos,
  yPos,
} from './stats.mjs';

// A fixed local midday timestamp per day avoids the test depending on the
// runner's timezone landing a UTC day-boundary on a different local day.
function localDay(y, m, d) {
  return new Date(y, m - 1, d, 12, 0, 0).getTime();
}

describe('liftOf', () => {
  it('defaults to squat for entries logged before the lift selector existed', () => {
    expect(liftOf({})).toBe('squat');
  });

  it('otherwise passes the lift through', () => {
    expect(liftOf({ lift: 'bench' })).toBe('bench');
  });
});

describe('dayKey', () => {
  it('buckets timestamps on the same local day identically', () => {
    const morning = localDay(2026, 3, 14) - 4 * 3600e3;
    const evening = localDay(2026, 3, 14) + 4 * 3600e3;
    expect(dayKey(morning)).toBe(dayKey(evening));
  });

  it('buckets different days differently', () => {
    expect(dayKey(localDay(2026, 3, 14))).not.toBe(dayKey(localDay(2026, 3, 15)));
  });
});

describe('aggregate', () => {
  it('returns nothing for an empty history', () => {
    expect(aggregate([])).toEqual([]);
  });

  it('sorts days oldest-first regardless of input order', () => {
    const list = [
      { type: 'competition', lift: 'squat', time: localDay(2026, 3, 15), verdict: 'GOOD' },
      { type: 'competition', lift: 'squat', time: localDay(2026, 3, 10), verdict: 'GOOD' },
      { type: 'competition', lift: 'squat', time: localDay(2026, 3, 12), verdict: 'GOOD' },
    ];
    const days = aggregate(list);
    expect(days.map((d) => d.time)).toEqual([
      localDay(2026, 3, 10),
      localDay(2026, 3, 12),
      localDay(2026, 3, 15),
    ]);
  });

  it('counts every rep verdict in a training set, and every competition attempt', () => {
    const list = [
      {
        type: 'training', lift: 'squat', time: localDay(2026, 3, 1),
        reps: [{ verdict: 'GOOD' }, { verdict: 'GOOD' }, { verdict: 'NO_LIFT' }, { verdict: 'UNCERTAIN' }],
      },
      { type: 'competition', lift: 'squat', time: localDay(2026, 3, 1), verdict: 'GOOD' },
    ];
    const [day] = aggregate(list);
    expect(day.good).toBe(3); // 2 training reps + 1 competition attempt
    expect(day.noLift).toBe(1);
    expect(day.uncertain).toBe(1);
  });

  it('merges same-day entries into one bucket', () => {
    const list = [
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1) + 1000, reps: [{ verdict: 'GOOD' }] },
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1) + 9999, reps: [{ verdict: 'NO_LIFT' }] },
    ];
    const days = aggregate(list);
    expect(days).toHaveLength(1);
    expect(days[0].good).toBe(1);
    expect(days[0].noLift).toBe(1);
  });

  it('converts lb to kg before comparing against the day\'s top weight', () => {
    // 225 lb is the heavier lift, but only once correctly converted (~102 kg)
    // does it actually beat the 100 kg training set logged the same day.
    const list = [
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1), weight: '100', unit: 'kg', reps: [] },
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1), weight: '225', unit: 'lb', reps: [] },
    ];
    const [day] = aggregate(list);
    expect(day.top.squat).toBeCloseTo(225 * 0.45359237, 6);
    expect(day.top.squat).toBeGreaterThan(100);
  });

  it('keeps the heaviest set per lift per day, not the last one logged', () => {
    const list = [
      { type: 'training', lift: 'bench', time: localDay(2026, 3, 1), weight: '80', unit: 'kg', reps: [] },
      { type: 'training', lift: 'bench', time: localDay(2026, 3, 1), weight: '60', unit: 'kg', reps: [] },
    ];
    const [day] = aggregate(list);
    expect(day.top.bench).toBe(80);
  });

  it('ignores an unparseable or non-positive weight rather than corrupting the max', () => {
    const list = [
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1), weight: 'not-a-number', reps: [] },
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1), weight: '0', reps: [] },
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1), weight: '-10', reps: [] },
    ];
    const [day] = aggregate(list);
    expect(day.top).toEqual({});
  });

  it('a competition attempt with no verdict yet does not count as a rep', () => {
    const list = [{ type: 'competition', lift: 'squat', time: localDay(2026, 3, 1), verdict: null }];
    const [day] = aggregate(list);
    expect(day.good + day.noLift + day.uncertain).toBe(0);
  });

  it('tracks top weight independently per lift within the same day', () => {
    const list = [
      { type: 'training', lift: 'squat', time: localDay(2026, 3, 1), weight: '140', unit: 'kg', reps: [] },
      { type: 'training', lift: 'bench', time: localDay(2026, 3, 1), weight: '90', unit: 'kg', reps: [] },
    ];
    const [day] = aggregate(list);
    expect(day.top).toEqual({ squat: 140, bench: 90 });
  });
});

describe('xPos / yPos', () => {
  it('a single point centers on the chart', () => {
    expect(xPos(0, 1)).toBeCloseTo((CHART_PAD.l + CHART_WIDTH - CHART_PAD.r) / 2, 6);
  });

  it('spans the full plot width for the first and last of many points', () => {
    expect(xPos(0, 5)).toBeCloseTo(CHART_PAD.l, 6);
    expect(xPos(4, 5)).toBeCloseTo(CHART_WIDTH - CHART_PAD.r, 6);
  });

  it('is monotonic in i', () => {
    const xs = [0, 1, 2, 3, 4].map((i) => xPos(i, 5));
    for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThan(xs[i - 1]);
  });

  it('yPos is inverted — max value sits at the top of the chart (smallest y)', () => {
    expect(yPos(100, 100)).toBeCloseTo(CHART_PAD.t, 6);
    expect(yPos(0, 100)).toBeCloseTo(CHART_HEIGHT - CHART_PAD.b, 6);
    expect(yPos(100, 100)).toBeLessThan(yPos(0, 100));
  });
});

describe('linePath', () => {
  it('starts with M and continues with L, one decimal place', () => {
    expect(linePath([[1.23456, 2], [3, 4.5]])).toBe('M1.2 2.0 L3.0 4.5');
  });

  it('is empty for no points, and a bare M for one', () => {
    expect(linePath([])).toBe('');
    expect(linePath([[5, 6]])).toBe('M5.0 6.0');
  });
});

describe('LIFT_COLORS', () => {
  it('has an entry for every lift the legend and chart series need', () => {
    expect(Object.keys(LIFT_COLORS).sort()).toEqual(['bench', 'deadlift', 'squat']);
  });
});
