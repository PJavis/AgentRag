import { test as setup, expect } from '@playwright/test'

// Logs in once via the real UI and saves the authenticated storage state so
// every spec can reuse it (no per-test login).
const authFile = 'e2e/.auth/user.json'

setup('authenticate', async ({ page }) => {
  await page.goto('/login')
  // When auth is disabled the /login page auto-redirects to /notebooks; when it
  // is enabled we log in via the form. Handle both: try the form, but tolerate
  // it being absent / detached by the redirect, then wait to land off /login.
  try {
    const password = page.locator('input[type="password"]')
    await password.waitFor({ state: 'visible', timeout: 3_000 })
    await page.locator('input[type="email"]').fill('e2e@test.local')
    await password.fill('e2epass123')
    await password.press('Enter')
  } catch {
    // form not present or page already redirecting (auth disabled) — fall through
  }
  await page.waitForURL((url) => !url.pathname.startsWith('/login'), { timeout: 20_000 })
  await page.context().storageState({ path: authFile })
})
