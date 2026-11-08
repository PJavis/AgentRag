import { describe, expect, it } from 'vitest'
import { citationPage, pageUrl } from './open-original'

describe('citationPage', () => {
  it('prefers page_start', () => {
    expect(citationPage({ page_start: 7, page: '7-9' })).toBe(7)
  })
  it('falls back to a numeric or range-string page', () => {
    expect(citationPage({ page: 3 })).toBe(3)
    expect(citationPage({ page: '12-14' })).toBe(12)
  })
  it('returns null without a usable page', () => {
    expect(citationPage({})).toBeNull()
    expect(citationPage({ page_start: 0 })).toBeNull()
    expect(citationPage({ page: 'n/a' })).toBeNull()
  })
})

describe('pageUrl', () => {
  it('adds the PDF page fragment', () => {
    expect(pageUrl('blob:x/1', 5)).toBe('blob:x/1#page=5')
    expect(pageUrl('blob:x/1', null)).toBe('blob:x/1')
  })
})
