import { test, expect } from '@playwright/test';

test('empty state shows before any history exists', async ({ page }) => {
  await page.goto('stats.html', { waitUntil: 'networkidle' });
  await expect(page.locator('#empty')).toBeVisible();
  await expect(page.locator('#content')).toBeHidden();
});

test('sample data preview renders real charts without writing to storage', async ({ page }) => {
  await page.goto('stats.html', { waitUntil: 'networkidle' });

  await page.click('#sampleToggle');
  await expect(page.locator('#sampleBanner')).toBeVisible();
  await expect(page.locator('#content')).toBeVisible();
  await expect(page.locator('#empty')).toBeHidden();

  // 28 days of generated sample data guarantees enough spread for every chart
  // to clear its own "need at least two days" threshold and draw something.
  await expect(page.locator('#goodRateChart svg')).toBeVisible();
  await expect(page.locator('#weightChart svg')).toBeVisible();
  await expect(page.locator('#volumeChart svg')).toBeVisible();

  // The whole point of "preview" — never actually persisted.
  const stored = await page.evaluate(() => localStorage.getItem('wl.history'));
  expect(stored).toBeNull();

  // Toggling off returns to the real (empty) history, not a stale render.
  await page.click('#sampleToggle');
  await expect(page.locator('#empty')).toBeVisible();
  await expect(page.locator('#sampleBanner')).toBeHidden();
});

test('real logged history renders without the sample banner', async ({ page }) => {
  await page.goto('stats.html', { waitUntil: 'commit' });
  const now = Date.now();
  await page.evaluate((now) => {
    const day = 86400000;
    localStorage.setItem('wl.history', JSON.stringify([
      { type: 'training', lift: 'squat', time: now, weight: '120', unit: 'kg',
        reps: [{ verdict: 'GOOD', time: now }] },
      { type: 'training', lift: 'squat', time: now - day, weight: '115', unit: 'kg',
        reps: [{ verdict: 'GOOD', time: now }, { verdict: 'NO_LIFT', time: now }] },
    ]));
  }, now);
  await page.reload({ waitUntil: 'networkidle' });

  await expect(page.locator('#content')).toBeVisible();
  await expect(page.locator('#sampleBanner')).toBeHidden();
  await expect(page.locator('#sumSessions')).toHaveText('2');
  await expect(page.locator('#sumReps')).toHaveText('3');
});
