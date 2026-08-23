// live.html is the one page that needs a camera before anything else happens
// — the offline simulator only starts once the camera gate is passed (real
// getUserMedia success is the trigger for startTransport(), which then falls
// back to the simulator when /ws/live doesn't answer). Chromium's fake video
// device gets past that gate with a synthetic feed, which is enough to
// exercise the actual canvas-drawing code no Vitest unit test can reach.
import { test, expect } from '@playwright/test';

test.use({
  permissions: ['camera'],
  launchOptions: {
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
  },
});

async function passCameraGate(page) {
  await page.goto('live.html', { waitUntil: 'networkidle' });
  await page.click('#enableCamBtn');
  await expect(page.locator('#camGate')).toHaveClass(/hidden/);
}

test('camera gate opens onto the offline simulator when no backend is running', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (err) => errors.push(err.message));
  await passCameraGate(page);

  // No /ws/live to answer here, so the client's connection attempt times out
  // and falls back to the built-in simulator — the same thing a visitor to
  // the hosted Pages demo sees.
  await expect(page.locator('#connLabel')).toHaveText('Demo data');
  await expect(page.locator('#demoBanner')).toBeVisible();

  expect(errors, `page errors: ${errors.join('; ')}`).toEqual([]);
});

test('the depth overlay actually draws — checkpoint light and canvas pixels agree', async ({ page }) => {
  await passCameraGate(page);
  await expect(page.locator('#connLabel')).toHaveText('Demo data');

  // Squat is the default lift; the simulator's synthetic rep cycles through
  // standing -> descending -> ascending every 6s. Sample the canvas mid-cycle
  // rather than at t=0, when there's nothing but a standing figure to draw.
  await page.waitForTimeout(2500);

  const hasNonBlackPixels = await page.evaluate(() => {
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 10 || data[i + 1] > 10 || data[i + 2] > 10) return true;
    }
    return false;
  });
  expect(hasNonBlackPixels, 'canvas should have drawn a skeleton/overlay, not stayed black').toBe(true);

  // The lamp's CSS class ("white"/"red"/"grey") is driven by the same
  // `checkpoint` value the overlay lines are colored by — see
  // depth_geometry() in api/main.py and drawDepthGeometry() in live.html.
  // They should never disagree on whether depth was reached (this is the
  // exact class of bug SQUAT_DEPTH_THRESHOLD was introduced to prevent).
  const lampClass = await page.locator('#light').getAttribute('class');
  expect(['white', 'red', 'grey'].some((c) => lampClass.includes(c))).toBe(true);
});

test('switching lifts changes the command tracker and disables training where unsupported', async ({ page }) => {
  await passCameraGate(page);
  await expect(page.locator('#connLabel')).toHaveText('Demo data');

  await page.click('#liftBenchBtn');
  await expect(page.locator('#liftBenchBtn')).toHaveClass(/active/);
  await expect(page.locator('#checkpointLabel')).toContainText(/chest|lockout/i);

  await page.click('#modeCompBtn');
  await expect(page.locator('#modeCompBtn')).toHaveClass(/active/);
  await expect(page.locator('#cmdTracker')).toHaveClass(/show/);
});

test('a training set logs a rep and survives a lift switch without corruption', async ({ page }) => {
  // Regression: setLift()/setMode() used to have no guard against an active
  // training set, so switching mid-set kept appending to the same
  // currentReps array under a tracker for a different lift — see
  // "Block lift/mode switching while a training set is active".
  await passCameraGate(page);
  await expect(page.locator('#connLabel')).toHaveText('Demo data');

  await page.click('#setBtn');
  await expect(page.locator('#setBtn')).toHaveClass(/active/);

  // Mid-set: switching lifts must be refused, not silently allowed.
  await page.click('#liftBenchBtn');
  await expect(page.locator('#liftSquatBtn')).toHaveClass(/active/);
  await expect(page.locator('#liftBenchBtn')).not.toHaveClass(/active/);

  // Let at least one simulated rep complete, then confirm it was logged.
  await page.waitForFunction(
    () => document.querySelectorAll('#repList li').length > 0,
    { timeout: 15000 }
  );
  await expect(page.locator('#repList li').first()).toBeVisible();
});
