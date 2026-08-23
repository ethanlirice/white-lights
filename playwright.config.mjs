// Real-browser tests for web/*.html — the layer web/lib/*.test.mjs (Vitest)
// cannot reach. Vitest tests the pure logic in isolation; these tests load the
// actual pages in Chromium and drive them the way a user would, which is the
// only way to exercise `<script type="module">` at all (jsdom does not
// execute module scripts; confirmed directly before reaching for this).
//
// `webServer` below serves web/ over plain HTTP with Python's stdlib server —
// no new dependency, and it's the same tool this project already uses to
// simulate the GitHub Pages deploy locally (see .github/workflows/pages.yml).
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  // Even a handful of real Chromium instances running concurrently was enough
  // to starve this machine — observed directly, twice, at two different
  // worker counts: the full suite flaked under both full default parallelism
  // (5 workers) and a workers:3 cap, on different tests each time, with page
  // snapshots at the moment of failure showing correctly-rendered data — a
  // resource-contention signature (something was too slow), not a real bug
  // (the assertion would have passed). Serialised rather than trusting a
  // parallelism number to be safe on hardware this variable; a generous
  // default timeout below covers the same contention for whichever action
  // happens to be running when the machine is briefly starved.
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'list' : 'html',
  timeout: 45_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: 'http://127.0.0.1:8073',
    trace: 'retain-on-failure',
    actionTimeout: 10_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'python3 -m http.server 8073 --directory web',
    url: 'http://127.0.0.1:8073/landing.html',
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
