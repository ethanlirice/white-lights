// Every page must load cleanly in a real browser: no thrown errors, no
// console.error, and its nav/theme-toggle wiring actually present. This is
// the baseline the deeper per-page specs build on.
import { test, expect } from '@playwright/test';

const PAGES = ['landing.html', 'live.html', 'history.html', 'stats.html', 'upload.html'];

for (const page of PAGES) {
  test(`${page} loads with no errors and a working theme toggle`, async ({ page: browserPage }) => {
    const errors = [];
    browserPage.on('pageerror', (err) => errors.push(err.message));
    browserPage.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await browserPage.goto(page, { waitUntil: 'networkidle' });

    // Theme toggle: every page has one, wired to the same localStorage key.
    // Exercising it is a cheap way to confirm the page's module script (where
    // one exists) actually executed and attached its listeners — a page whose
    // script silently failed to load would still render its static HTML.
    const toggle = browserPage.locator('#themeToggle');
    await expect(toggle).toBeVisible();
    const before = await browserPage.evaluate(() => document.documentElement.getAttribute('data-theme'));
    await toggle.click();
    const after = await browserPage.evaluate(() => document.documentElement.getAttribute('data-theme'));
    expect(after).not.toBe(before);
    expect(await browserPage.evaluate(() => localStorage.getItem('wl.theme'))).toBe(
      after === 'light' ? 'light' : 'dark'
    );

    expect(errors, `console/page errors on ${page}: ${errors.join('; ')}`).toEqual([]);
  });
}
