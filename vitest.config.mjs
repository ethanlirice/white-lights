// Scopes Vitest to the pure-logic unit tests in web/lib/. Needed explicitly
// once e2e/ existed: Vitest's own default include pattern also matches
// *.spec.mjs, which is what the Playwright specs in e2e/ are named — without
// this, `npm test` tried to run them as Vitest tests too, and they failed
// (they use @playwright/test's `test`/`expect`, which needs a real browser
// Vitest's workers don't have) with worker-timeout errors that had nothing
// to do with the actual specs. `npm run test:e2e` (Playwright) owns e2e/.
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['web/lib/**/*.test.mjs'],
  },
});
