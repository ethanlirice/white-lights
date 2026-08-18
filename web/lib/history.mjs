/**
 * Merging an imported history export back into local storage.
 *
 * `live.html` writes new entries with `unshift`, so the stored array is an
 * invariant every reader depends on: newest first. `history.html`'s
 * `render()` groups entries by day in the order they first appear in the
 * array — it does not re-sort by time — so that invariant isn't optional,
 * it's load-bearing for the page to render chronologically at all.
 *
 * Both pages' "Import" buttons used to rebuild the array with a plain
 * `incoming.concat(existing)` and save that directly. Concatenating two
 * newest-first arrays does not produce a newest-first array unless every
 * entry in one array is older than every entry in the other — false for the
 * ordinary case of importing a real export spanning multiple sessions.
 * `mergeHistory` is the one place that invariant gets restored, used
 * identically by both pages instead of two copies quietly drifting.
 */

/**
 * Merge an imported history array with the existing one, newest first.
 *
 * @throws {Error} if `incoming` is not an array — the same validation both
 *   pages performed inline before this existed, centralised so the error
 *   message can't drift between them either.
 */
export function mergeHistory(incoming, existing) {
  if (!Array.isArray(incoming)) {
    throw new Error('not an array');
  }
  // `|| 0` treats a missing/malformed `time` as the epoch rather than
  // `undefined - undefined` (NaN) — sorts such entries to the end instead of
  // leaving the comparator's behaviour on them technically unspecified.
  return incoming.concat(existing).sort((a, b) => (b.time || 0) - (a.time || 0));
}
