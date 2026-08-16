import { describe, expect, it } from 'vitest';
import {
  SQUAT_DEPTH_THRESHOLD,
  jitter,
  synthBenchPose,
  synthDeadliftPose,
  synthPose,
  synthSquatPose,
  toKeypointList,
} from './poses.mjs';

const EXPECTED_JOINTS = [
  'left_shoulder', 'right_shoulder',
  'left_hip', 'right_hip',
  'left_knee', 'right_knee',
  'left_ankle', 'right_ankle',
  'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist',
];

// A representative sweep of the simulator's own `t`, not just t=0 — jitter is
// a function of t, so a bug that only shows up once the sine wave has moved
// would survive a single fixed timestamp.
const TIMES = [0, 0.7, 1.3, 4.1, 12.9];

function byName(keypoints) {
  return Object.fromEntries(keypoints.map((kp) => [kp.name, kp]));
}

describe('toKeypointList', () => {
  it('turns a name -> {x, y} map into named points with a fixed confidence', () => {
    const list = toKeypointList({ a: { x: 1, y: 2 }, b: { x: 3, y: 4 } });
    expect(list).toEqual([
      { name: 'a', x: 1, y: 2, confidence: 0.9 },
      { name: 'b', x: 3, y: 4, confidence: 0.9 },
    ]);
  });

  it('produces nothing from an empty pose', () => {
    expect(toKeypointList({})).toEqual([]);
  });
});

describe('jitter', () => {
  it('is deterministic given the same (t, seed)', () => {
    expect(jitter(3.14, 2)).toBe(jitter(3.14, 2));
  });

  it('stays within its documented amplitude', () => {
    for (const t of [0, 1, 2, 3, 100]) {
      for (const seed of [0, 1, 2, 3, 4]) {
        expect(Math.abs(jitter(t, seed))).toBeLessThanOrEqual(0.0035 + 1e-12);
      }
    }
  });
});

describe.each([
  ['synthSquatPose', synthSquatPose],
  ['synthBenchPose', synthBenchPose],
  ['synthDeadliftPose', synthDeadliftPose],
])('%s', (_name, fn) => {
  it('emits every joint the skeleton overlay draws', () => {
    for (const t of TIMES) {
      const { keypoints } = fn(t, 0.5);
      const names = keypoints.map((kp) => kp.name).sort();
      expect(names).toEqual([...EXPECTED_JOINTS].sort());
    }
  });

  it('is deterministic given the same (t, progress)', () => {
    const a = fn(1.5, 0.4);
    const b = fn(1.5, 0.4);
    expect(a).toEqual(b);
  });

  it('every confidence is the fixed 0.9 the overlay treats as "visible"', () => {
    const { keypoints } = fn(0, 0.5);
    for (const kp of keypoints) expect(kp.confidence).toBe(0.9);
  });

  it('every coordinate is a finite number', () => {
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      const { keypoints } = fn(2.2, p);
      for (const kp of keypoints) {
        expect(Number.isFinite(kp.x)).toBe(true);
        expect(Number.isFinite(kp.y)).toBe(true);
      }
    }
  });
});

describe('synthSquatPose geometry', () => {
  it('is the only pose that reports geometry (the others do not judge depth)', () => {
    expect(synthSquatPose(0, 0.5).geometry).not.toBeNull();
    expect(synthBenchPose(0, 0.5).geometry).toBeNull();
    expect(synthDeadliftPose(0, 0.5).geometry).toBeNull();
  });

  it('margin and below always agree on sign', () => {
    for (const t of TIMES) {
      for (let p = 0; p <= 1; p += 0.05) {
        const { geometry } = synthSquatPose(t, p);
        expect(geometry.below).toBe(geometry.margin > 0);
      }
    }
  });

  it('the geometry rows are the same rows the keypoints draw the hip/knee at', () => {
    // The overlay trusts `geometry.{hip_row,knee_row}` as ground truth for
    // where to draw its lines — they must actually be the hip/knee keypoints'
    // y-coordinates, not a separate number that happens to usually agree.
    for (const t of TIMES) {
      for (const p of [0, 0.3, 0.55, 0.8, 1]) {
        const { keypoints, geometry } = synthSquatPose(t, p);
        const kp = byName(keypoints);
        const hipY = (kp.left_hip.y + kp.right_hip.y) / 2;
        const kneeY = (kp.left_knee.y + kp.right_knee.y) / 2;
        expect(geometry.hip_row).toBeCloseTo(hipY, 10);
        expect(geometry.knee_row).toBeCloseTo(kneeY, 10);
      }
    }
  });

  it('standing (progress=0) is always above parallel, regardless of jitter', () => {
    for (const t of TIMES) {
      expect(synthSquatPose(t, 0).geometry.below).toBe(false);
    }
  });

  it('the bottom of the descent (progress=1) is always below parallel', () => {
    for (const t of TIMES) {
      expect(synthSquatPose(t, 1).geometry.below).toBe(true);
    }
  });

  it('crosses parallel close to the same progress the checkpoint light uses', () => {
    // This is the regression this file exists for: live.html's trainingStatus
    // and competitionStatus light the depth checkpoint at
    // `progress > SQUAT_DEPTH_THRESHOLD`. If the drawn geometry crosses at a
    // meaningfully different point, the light and the lines disagree with
    // each other on screen for the gap between the two thresholds — which is
    // exactly what shipped before this constant was tuned and shared: the
    // geometry used to cross at progress ~1.02, entirely outside [0, 1], so
    // the light went white and the lines never agreed for the whole demo.
    //
    // The offset here (0.05) is comfortably past the trajectory's own jitter
    // noise (worst case ~0.0034 across TIMES, i.e. under a 0.03 progress
    // offset) — this asserts the crossing is *near* the threshold, not that
    // jitter never perturbs it by a thousandth.
    for (const t of TIMES) {
      const justBelow = synthSquatPose(t, SQUAT_DEPTH_THRESHOLD - 0.05);
      const justAbove = synthSquatPose(t, SQUAT_DEPTH_THRESHOLD + 0.05);
      expect(justBelow.geometry.below).toBe(false);
      expect(justAbove.geometry.below).toBe(true);
    }
  });

  it('never draws the hip below the ankle, however deep the demo rep goes', () => {
    for (const t of TIMES) {
      const { keypoints } = synthSquatPose(t, 1);
      const kp = byName(keypoints);
      expect(kp.left_hip.y).toBeLessThan(kp.left_ankle.y);
      expect(kp.right_hip.y).toBeLessThan(kp.right_ankle.y);
    }
  });
});

describe('synthPose', () => {
  it('dispatches by lift name', () => {
    expect(synthPose('bench', 0, 0.5)).toEqual(synthBenchPose(0, 0.5));
    expect(synthPose('deadlift', 0, 0.5)).toEqual(synthDeadliftPose(0, 0.5));
    expect(synthPose('squat', 0, 0.5)).toEqual(synthSquatPose(0, 0.5));
  });

  it('falls back to squat for an unrecognised lift name', () => {
    // trainingStatus/competitionStatus only ever pass a name the UI itself
    // offers, but a silent fallback here is safer than a silent `undefined`.
    expect(synthPose('deadlift-typo', 0, 0.5)).toEqual(synthSquatPose(0, 0.5));
  });
});
