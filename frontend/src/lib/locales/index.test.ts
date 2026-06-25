import { describe, it, expect } from 'vitest'
import fs from 'fs'
import path from 'path'
import { resources } from './index'
import { enUS } from './en-US'

const getKeys = (obj: Record<string, unknown>, prefix = ''): string[] => {
  return Object.keys(obj).reduce((res: string[], el) => {
    const val = obj[el]
    if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
      return [...res, ...getKeys(val as Record<string, unknown>, prefix + el + '.')]
    }
    return [...res, prefix + el]
  }, [])
}

describe('Locale Parity', () => {
  const enKeys = getKeys(enUS)

  const locales = Object.entries(resources).filter(([code]) => code !== 'en-US')

  it.each(locales.map(([code, resource]) => [code, resource] as const))(
    '%s should have the same keys as en-US',
    (code, resource) => {
      const localeKeys = getKeys(resource.translation as Record<string, unknown>)

      const missing = enKeys.filter(key => !localeKeys.includes(key))
      const extra = localeKeys.filter(key => !enKeys.includes(key))

      expect(missing, `Missing keys in ${code}: ${missing.join(', ')}`).toEqual([])
      expect(extra, `Extra keys in ${code}: ${extra.join(', ')}`).toEqual([])
    },
  )
})

describe('Unused Key Detection', () => {
  it(
    'all en-US leaf keys should be referenced in source files',
    () => {
      const srcDir = path.resolve(__dirname, '../../..')
      const localesDir = path.resolve(__dirname)

      const files = fs.readdirSync(srcDir, { recursive: true }) as string[]
      const sourceFiles = files.filter(f => {
        const full = path.join(srcDir, f)
        if (full.startsWith(localesDir)) return false
        if (f.endsWith('.test.ts') || f.endsWith('.test.tsx')) return false
        return f.endsWith('.ts') || f.endsWith('.tsx')
      })

      // Normalize optional chaining (t?.common?.key → t.common.key)
      // so that keys like "common.errorDetails" match "common?.errorDetails"
      const corpus = sourceFiles
        .map(f => fs.readFileSync(path.join(srcDir, f), 'utf-8'))
        .join('\n')
        .replace(/\?\./g, '.')

      // A key is "referenced" if its full dotted path OR a meaningful ancestor
      // (section.subsection or deeper) appears in source — covers objects fetched
      // whole, e.g. t('sources.ingestStages', { returnObjects: true }) then indexed
      // by stage. We stop at depth 2 so a bare top-level section ('common') can't
      // mark everything used.
      const isReferenced = (key: string): boolean => {
        const parts = key.split('.')
        for (let i = parts.length; i >= 2; i--) {
          if (corpus.includes(parts.slice(0, i).join('.'))) return true
        }
        return false
      }

      // Keys with no current string-literal reference in source — either dead (the
      // UI hardcoded the English) or pending wiring. Tracked here so the gate stays
      // green; when deleting a key, remove it from the locales AND this list.
      const KNOWN_UNREFERENCED = new Set([
        'auth.connectErrorHint', 'auth.loginDesc', 'auth.loginTitle',
        'auth.passwordPlaceholder', 'auth.signIn', 'auth.signingIn',
        'common.apiUrl', 'common.built', 'common.checkConsoleLogs',
        'common.connectionError', 'common.diagnosticInfo', 'common.frontendUrl',
        'common.retryConnection', 'common.unableToConnect', 'common.version',
      ])

      const leafKeys = getKeys(enUS)
      const unused = leafKeys.filter(key => !isReferenced(key) && !KNOWN_UNREFERENCED.has(key))

      expect(
        unused,
        `Found ${unused.length} unused i18n key(s):\n${unused.join('\n')}`,
      ).toEqual([])
    },
    30_000,
  )
})
