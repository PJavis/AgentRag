import { test, expect } from '@playwright/test'

// The navigation fix: cost / activity / admin-activity pages now wrap in
// AppShell, so the sidebar (= the way "home") is present. (Auth via storageState.)

const PAGES = [
  { path: '/cost', name: 'cost' },
  { path: '/activity', name: 'activity' },
  { path: '/admin/activity', name: 'admin-activity' },
]

for (const p of PAGES) {
  test(`${p.name} renders the sidebar (home nav present)`, async ({ page }) => {
    await page.goto(p.path)
    await expect(
      page.locator('a[href="/notebooks"]').first(),
      'sidebar home/notebooks link should be visible',
    ).toBeVisible()
    await expect(page.locator('a[href="/cost"]').first()).toBeVisible()
    await expect(page.locator('a[href="/activity"]').first()).toBeVisible()
    await page.screenshot({ path: `e2e/screenshots/${p.name}.png`, fullPage: true })
  })
}
