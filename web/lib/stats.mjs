/**
 * Pure aggregation + chart-geometry math behind web/stats.html.
 *
 * Extracted for the same reason as web/lib/poses.mjs: this is real
 * arithmetic — unit conversion, day-bucketing, verdict counting, percentile
 * positioning — sitting in a page with no test runner. `aggregate()` in
 * particular is the one function every number on the stats page passes
 * through; a bug here would misreport a lifter's own logged history back to
 * them silently. See stats.mjs.test.mjs.
 *
 * SVG-rendering helpers (frame(), xLabels(), yLabels(), the render* functions)
 * stay inline in stats.html — they touch `document.createElementNS` and have
 * nothing to assert on without a DOM, so extracting them would trade "hard to
 * test" for "expensive to test" rather than removing the problem. The
 * *inputs* those functions receive from the code below are what needed
 * coverage, not the drawing itself.
 */

/** `e.lift` defaults to squat — entries logged before the lift selector existed. */
export function liftOf(entry) {
  return entry.lift || 'squat';
}

/** Local-calendar-day bucket key. Not zero-padded — it's a Map key, never displayed. */
export function dayKey(t) {
  const d = new Date(t);
  return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
}

/** Short display label for a day, e.g. "Aug 14". */
export function dayShort(t) {
  return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

const LB_TO_KG = 0.45359237;

/**
 * Bucket a flat history list into one summary per calendar day, sorted
 * oldest-first (the charts read left-to-right in time).
 *
 * Each day: { time, good, noLift, uncertain, top: {lift: heaviestKgThatDay} }.
 * Weight is always normalised to kg here — everything downstream can assume
 * one unit, rather than each chart re-deriving the lb/kg check.
 */
export function aggregate(list) {
  const days = {};
  list.forEach((e) => {
    const k = dayKey(e.time);
    if (!days[k]) days[k] = { time: e.time, good: 0, noLift: 0, uncertain: 0, top: {} };
    const d = days[k];
    const count = (v) => {
      if (v === 'GOOD') d.good++;
      else if (v === 'NO_LIFT') d.noLift++;
      else d.uncertain++;
    };
    if (e.type === 'training') {
      (e.reps || []).forEach((r) => count(r.verdict));
      const w = parseFloat(e.weight);
      if (!isNaN(w) && w > 0) {
        const kg = e.unit === 'lb' ? w * LB_TO_KG : w;
        const l = liftOf(e);
        if (!d.top[l] || kg > d.top[l]) d.top[l] = kg;
      }
    } else if (e.verdict) {
      count(e.verdict);
    }
  });
  return Object.keys(days).map((k) => days[k]).sort((a, b) => a.time - b.time);
}

export const LIFT_COLORS = {
  squat: 'oklch(70% 0.14 250)',
  bench: 'oklch(70% 0.14 85)',
  deadlift: 'oklch(70% 0.14 150)',
};

// The SVG viewBox every chart shares — kept here (not just in stats.html) so
// xPos/yPos, which are pure functions of it, can be tested against the exact
// values the page actually renders with, not a copy that could drift.
export const CHART_WIDTH = 860;
export const CHART_HEIGHT = 220;
export const CHART_PAD = { l: 38, r: 12, t: 12, b: 26 };

/** Horizontal pixel position for point `i` of `n`, evenly spaced inside the pad. */
export function xPos(i, n) {
  if (n === 1) return (CHART_PAD.l + CHART_WIDTH - CHART_PAD.r) / 2;
  return CHART_PAD.l + (CHART_WIDTH - CHART_PAD.l - CHART_PAD.r) * i / (n - 1);
}

/** Vertical pixel position for value `v` against axis max `max` (SVG y grows down). */
export function yPos(v, max) {
  return CHART_PAD.t + (CHART_HEIGHT - CHART_PAD.t - CHART_PAD.b) * (1 - v / max);
}

/** SVG path `d` for a polyline through `pts` ([x, y] pairs), one decimal place. */
export function linePath(pts) {
  return pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
}
