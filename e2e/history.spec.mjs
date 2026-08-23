// history.html reads/writes localStorage directly and groups entries by day
// in array order rather than re-sorting — see web/lib/history.mjs's header
// for why that made the Import button's ordering a real, previously-shipped
// bug. This exercises the actual rendering and filtering against realistic
// seeded data, in a real browser, rather than just the merge function alone
// (which web/lib/history.test.mjs already covers directly).
import { test, expect } from '@playwright/test';

const DAY_MS = 86400000;

function seedHistory(page, now) {
  return page.evaluate(({ now, day }) => {
    const seed = [
      { type: 'training', lift: 'squat', time: now, weight: '140', unit: 'kg',
        reps: [{ verdict: 'GOOD', time: now }] },
      { type: 'competition', lift: 'bench', time: now - day, verdict: 'NO_LIFT',
        faults: ['BAR_NOT_TO_CHEST'] },
      { type: 'training', lift: 'deadlift', time: now - 2 * day, weight: '180', unit: 'kg',
        reps: [{ verdict: 'GOOD', time: now }, { verdict: 'UNCERTAIN', time: now }] },
    ];
    localStorage.setItem('wl.history', JSON.stringify(seed));
  }, { now, day: DAY_MS });
}

test('renders seeded history newest-first, grouped by day, with correct summary counts', async ({ page }) => {
  const now = Date.now();
  await page.goto('history.html', { waitUntil: 'commit' });
  await seedHistory(page, now);
  await page.reload({ waitUntil: 'networkidle' });

  await expect(page.locator('.day-label')).toHaveText(['Today', 'Yesterday', /\w+day, /]);
  await expect(page.locator('.entry')).toHaveCount(3);

  // squat + deadlift are training sets; bench is the competition attempt.
  await expect(page.locator('#sumSets')).toHaveText('2');
  // 1 squat rep + 2 deadlift reps.
  await expect(page.locator('#sumReps')).toHaveText('3');
  await expect(page.locator('#sumAttempts')).toHaveText('1');
  // 2 GOOD out of 4 total judged (1 squat + 2 deadlift + 1 competition) = 50%.
  await expect(page.locator('#sumGoodRate')).toHaveText('50%');
});

test('lift filter narrows the list without touching the underlying data', async ({ page }) => {
  const now = Date.now();
  await page.goto('history.html', { waitUntil: 'commit' });
  await seedHistory(page, now);
  await page.reload({ waitUntil: 'networkidle' });

  await page.click('#liftFilter [data-v="bench"]');
  await expect(page.locator('.entry')).toHaveCount(1);
  await expect(page.locator('.entry-title')).toContainText('Bench');

  await page.click('#liftFilter [data-v="all"]');
  await expect(page.locator('.entry')).toHaveCount(3);
});

test('importing a multi-day export merges newest-first, not as two concatenated runs', async ({ page }) => {
  // The regression this file exists to guard: naive `incoming.concat(existing)`
  // preserved each array's own order but did not interleave them by time, so
  // day groups rendered out of chronological order after any real import.
  const now = Date.now();
  await page.goto('history.html', { waitUntil: 'commit' });

  // Existing: only today's set.
  await page.evaluate((now) => {
    localStorage.setItem('wl.history', JSON.stringify([
      { type: 'training', lift: 'squat', time: now, weight: '100', unit: 'kg',
        reps: [{ verdict: 'GOOD', time: now }] },
    ]));
  }, now);
  await page.reload({ waitUntil: 'networkidle' });

  // Import: two older entries, themselves already newest-first internally —
  // exactly the shape a real exported file has.
  const importPayload = JSON.stringify([
    { type: 'training', lift: 'bench', time: now - 3 * DAY_MS, weight: '80', unit: 'kg',
      reps: [{ verdict: 'GOOD', time: now }] },
    { type: 'training', lift: 'deadlift', time: now - 5 * DAY_MS, weight: '150', unit: 'kg',
      reps: [{ verdict: 'GOOD', time: now }] },
  ]);
  await page.setInputFiles('#importFile', {
    name: 'export.json',
    mimeType: 'application/json',
    buffer: Buffer.from(importPayload),
  });

  await expect(page.locator('.entry')).toHaveCount(3);
  // Today's squat set must still be the first entry rendered — if the import
  // had been concatenated naively, the two older imported entries (added via
  // `incoming.concat(existing)`) would sit ahead of it instead.
  await expect(page.locator('.entry-title').first()).toContainText('Squat');
});
