import { test as setup, expect } from '@playwright/test'

// Logs in once via the real UI and saves the authenticated storage state so
// every spec can reuse it (no per-test login).
const authFile = 'e2e/.auth/user.json'

setup('authenticate', async ({ page }) => {
  await page.goto('/login')
  await page.locator('input[type="email"]').fill('e2e@test.local')
  await page.locator('input[type="password"]').fill('e2epass123')
  await page.locator('button[type="submit"]').first().click()
  await page.waitForURL((url) => !url.pathname.startsWith('/login'), { timeout: 20_000 })
  await page.context().storageState({ path: authFile })
})
