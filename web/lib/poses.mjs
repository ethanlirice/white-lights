/**
 * Synthetic skeletons for the offline simulator (web/live.html).
 *
 * Extracted into a real module — rather than staying inline in live.html —
 * for one reason: this repo has no browser test runner, so this logic used to
 * be verified by reading it very carefully and counting braces by hand. It is
 * pure (no DOM, no globals, deterministic given its inputs), which makes "no
 * automated coverage" an unforced error rather than a constraint. See
 * poses.test.mjs.
 *
 * Loaded by the browser as a native ES module (`<script type="module">`) —
 * no bundler, no build step, nothing added to what ships. Imported the same
 * way by the test runner, so there is exactly one implementation to trust.
 *
 * Without these, the simulator sent `keypoints: null`, so the depth-geometry
 * overlay — the thing that makes this UI worth screenshotting — never
 * appeared on the hosted demo, the one place most visitors see it. These
 * fabricate a plausible body, not a plausible *lift*: good enough to show the
 * overlay moving in step with the state machine, not a physics sim.
 *
 * All three share one convention for `progress`, matching what
 * trainingStatus/competitionStatus already compute per lift, so the pose
 * functions can consume it directly with no re-mapping:
 *   squat     0 = standing, 1 = bottom of the descent
 *   bench     0 = lockout (bar away from chest), 1 = bar at the chest
 *   deadlift  0 = bar on the floor, 1 = standing lockout
 */

/**
 * The squat's depth checkpoint (the light going white, and the command
 * scripts' `cpAt`) and its depth-geometry overlay (the hip/knee lines, and
 * `geometry.below`) used to be two independent computations — live.html
 * thresholded `progress` directly, while this module derived `below` from
 * hip/knee Y-coordinates that, worked out numerically, never actually
 * crossed within `progress`'s [0, 1] range. The result: for the entire
 * "checkpoint met" portion of every demo rep, the light said depth was
 * confirmed while the lines it was supposedly reporting on stayed drawn as
 * insufficient — a standing self-contradiction on the one page most visitors
 * see. `synthSquatPose`'s hip trajectory is now tuned so the lines cross
 * exactly here; live.html imports this constant rather than hardcoding its
 * own copy of the number, so the two cannot drift apart again the same way.
 */
export const SQUAT_DEPTH_THRESHOLD = 0.55;

/** Idle micro-sway, not a rep — keeps the resting figure from looking frozen. */
export function jitter(t, seed) {
  return Math.sin(t * 2.3 + seed) * 0.0035;
}

export function toKeypointList(points) {
  const out = [];
  for (const name in points) {
    out.push({ name, x: points[name].x, y: points[name].y, confidence: 0.9 });
  }
  return out;
}

// Tuned so hipY(SQUAT_DEPTH_THRESHOLD) == kneeY(SQUAT_DEPTH_THRESHOLD) with the
// knee trajectory below (0.745 + 0.02p) — see SQUAT_DEPTH_THRESHOLD's comment.
// Concave (exponent < 1): the hip descends quickly at first and eases off
// near the bottom, so it clears the knee line by the checkpoint without
// overshooting the ankle by progress = 1.
const _HIP_DESCENT_TRAVEL = 0.2998;
const _HIP_DESCENT_EXPONENT = 0.4;

export function synthSquatPose(t, progress) {
  const cx = 0.5 + jitter(t, 4) * 0.4;
  const shoulderY = 0.30 + progress * 0.10 + jitter(t, 0);
  const hipY = 0.52 + _HIP_DESCENT_TRAVEL * Math.pow(progress, _HIP_DESCENT_EXPONENT) + jitter(t, 1);
  const kneeY = 0.745 + progress * 0.02 + jitter(t, 2);
  const ankleY = 0.93 + jitter(t, 3) * 0.3;
  const kneeForward = progress * 0.035; // knees travel forward on the descent
  const points = {
    left_shoulder: { x: cx - 0.085, y: shoulderY }, right_shoulder: { x: cx + 0.085, y: shoulderY },
    left_hip: { x: cx - 0.075, y: hipY }, right_hip: { x: cx + 0.075, y: hipY },
    left_knee: { x: cx - 0.065 + kneeForward, y: kneeY }, right_knee: { x: cx + 0.065 + kneeForward, y: kneeY },
    left_ankle: { x: cx - 0.05, y: ankleY }, right_ankle: { x: cx + 0.05, y: ankleY },
    left_elbow: { x: cx - 0.105, y: shoulderY + 0.02 }, right_elbow: { x: cx + 0.105, y: shoulderY + 0.02 },
    left_wrist: { x: cx - 0.10, y: shoulderY - 0.03 }, right_wrist: { x: cx + 0.10, y: shoulderY - 0.03 },
  };
  // Same two rows the real judge compares, from the same body — mirrors
  // `depth_geometry()` in api/main.py so the demo overlay isn't a special case.
  const below = hipY > kneeY;
  return {
    keypoints: toKeypointList(points),
    geometry: { hip_row: hipY, knee_row: kneeY, margin: (hipY - kneeY) * 480, below, confidence: 0.9 },
  };
}

export function synthBenchPose(t, progress) {
  const cx = 0.5 + jitter(t, 4) * 0.15;
  const shoulderY = 0.52 + jitter(t, 0) * 0.3;
  const hipY = 0.565 + jitter(t, 1) * 0.3;
  const kneeY = 0.60 + jitter(t, 2) * 0.3;
  const ankleY = 0.64 + jitter(t, 3) * 0.3;
  const barY = shoulderY - 0.20 + progress * 0.24; // lockout above the chest -> touch
  const elbowSpread = 0.05 + progress * 0.09;       // elbows flare as the bar descends
  const points = {
    left_shoulder: { x: cx - 0.09, y: shoulderY }, right_shoulder: { x: cx + 0.09, y: shoulderY },
    left_hip: { x: cx - 0.07, y: hipY }, right_hip: { x: cx + 0.07, y: hipY },
    left_knee: { x: cx - 0.10, y: kneeY }, right_knee: { x: cx + 0.10, y: kneeY },
    left_ankle: { x: cx - 0.09, y: ankleY }, right_ankle: { x: cx + 0.09, y: ankleY },
    left_elbow: { x: cx - 0.09 - elbowSpread, y: (shoulderY + barY) / 2 },
    right_elbow: { x: cx + 0.09 + elbowSpread, y: (shoulderY + barY) / 2 },
    left_wrist: { x: cx - 0.11, y: barY }, right_wrist: { x: cx + 0.11, y: barY },
  };
  return { keypoints: toKeypointList(points), geometry: null }; // bench does not judge depth
}

export function synthDeadliftPose(t, progress) {
  const cx = 0.5 + jitter(t, 4) * 0.25;
  const shoulderY = 0.40 - progress * 0.12 + jitter(t, 0);
  const hipY = 0.62 - progress * 0.12 + jitter(t, 1);
  const kneeY = 0.78 - progress * 0.04 + jitter(t, 2);
  const ankleY = 0.93 + jitter(t, 3) * 0.3;
  const barY = ankleY - progress * (ankleY - hipY); // bar rides the shin up to hip height
  const points = {
    left_shoulder: { x: cx - 0.08, y: shoulderY }, right_shoulder: { x: cx + 0.08, y: shoulderY },
    left_hip: { x: cx - 0.07, y: hipY }, right_hip: { x: cx + 0.07, y: hipY },
    left_knee: { x: cx - 0.06, y: kneeY }, right_knee: { x: cx + 0.06, y: kneeY },
    left_ankle: { x: cx - 0.045, y: ankleY }, right_ankle: { x: cx + 0.045, y: ankleY },
    left_elbow: { x: cx - 0.09, y: (shoulderY + barY) / 2 }, right_elbow: { x: cx + 0.09, y: (shoulderY + barY) / 2 },
    left_wrist: { x: cx - 0.02, y: barY }, right_wrist: { x: cx + 0.02, y: barY },
  };
  return { keypoints: toKeypointList(points), geometry: null }; // deadlift does not judge depth
}

export function synthPose(liftName, t, progress) {
  if (liftName === 'bench') return synthBenchPose(t, progress);
  if (liftName === 'deadlift') return synthDeadliftPose(t, progress);
  return synthSquatPose(t, progress);
}
