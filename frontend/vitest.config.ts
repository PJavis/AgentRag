import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  css: { postcss: { plugins: [] } },
  test: {
    environment: 'jsdom',
    globals: true,
    // Only collect unit tests under src/. The Playwright specs in e2e/*.spec.ts
    // are run by Playwright, not vitest — excluding them avoids mis-collection.
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    setupFiles: ['./src/test/setup.ts'],
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  }
})
