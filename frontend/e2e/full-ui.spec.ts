import { test, expect } from '@playwright/test'

// Full-feature UI smoke: every primary route loads inside the AppShell, renders
// its own heading, does not redirect to /login, and does not hit the error
// boundary. Plus a notebook-detail check (seeded via API). Auth via storageState.

const API = 'http://localhost:8000/on/api'

const ROUTES = [
  { path: '/notebooks', name: 'notebooks' },
  { path: '/sources', name: 'sources' },
  { path: '/search', name: 'search' },
  { path: '/podcasts', name: 'podcasts' },
  { path: '/transformations', name: 'transformations' },
  { path: '/settings', name: 'settings' },
  { path: '/settings/api-keys', name: 'settings-api-keys' },
  { path: '/advanced', name: 'advanced' },
  { path: '/cost', name: 'cost' },
  { path: '/activity', name: 'activity' },
  { path: '/admin/activity', name: 'admin-activity' },
]

for (const r of ROUTES) {
  test(`route ${r.name} loads in-shell, renders, no crash`, async ({ page }) => {
    const consoleErrors: string[] = []
    page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()))

    await page.goto(r.path)

    // Not bounced to login.
    await expect(page).not.toHaveURL(/\/login/)
    // AppShell mounted (sidebar present).
    await expect(page.locator('a[href="/notebooks"]').first()).toBeVisible()
    // Page rendered a heading of its own.
    await expect(page.locator('h1, h2').first()).toBeVisible()
    // Error boundary not shown (English + Vietnamese fallback copy).
    await expect(page.getByText(/Something went wrong|đã xảy ra lỗi/i)).toHaveCount(0)

    await page.screenshot({ path: `e2e/screenshots/${r.name}.png`, fullPage: true })
  })
}

test('notebook detail page renders chat workspace', async ({ page, request }) => {
  // Seed a notebook via API (token from a fresh login).
  const login = await request.post(`${API}/auth/login`, {
    data: { email: 'e2e@test.local', password: 'e2epass123' },
  })
  const token = (await login.json()).access_token
  const created = await request.post(`${API}/notebooks`, {
    headers: { Authorization: `Bearer ${token}` },
    data: { name: 'E2E Detail NB', description: 'e2e' },
  })
  expect(created.ok()).toBeTruthy()
  const id = (await created.json()).id

  await page.goto(`/notebooks/${id}`)
  await expect(page).not.toHaveURL(/\/login/)
  await expect(page.locator('a[href="/notebooks"]').first()).toBeVisible()
  // Notebook name surfaced + a text input/area for chat is present.
  await expect(page.getByText('E2E Detail NB').first()).toBeVisible({ timeout: 15_000 })
  await expect(page.locator('textarea, input[type="text"]').first()).toBeVisible()
  await page.screenshot({ path: 'e2e/screenshots/notebook-detail.png', fullPage: true })
})
