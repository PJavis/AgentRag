import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { createCompactReferenceHoverComponent } from './source-references'
import type { Citation } from '@/lib/types/api'

vi.mock('@/lib/hooks/use-translation', () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}))

const cite = (over: Partial<Citation> = {}): Citation => ({
  document_title: 'Lec 10', source: 1, page_start: 7, ...over,
})

describe('citation badge click', () => {
  it('opens the cited page via onCitationClick when the citation is linked', () => {
    const onCitation = vi.fn()
    const onRef = vi.fn()
    const Link = createCompactReferenceHoverComponent([cite({ source_id: 'doc-1' })], onRef, onCitation)
    render(<Link href="#ref-c-1">1</Link>)
    fireEvent.click(screen.getByRole('button'))
    expect(onCitation).toHaveBeenCalledWith(expect.objectContaining({ source_id: 'doc-1', page_start: 7 }))
    expect(onRef).not.toHaveBeenCalled()
  })

  it('is not clickable without a source_id (nothing to open)', () => {
    const onCitation = vi.fn()
    const Link = createCompactReferenceHoverComponent([cite()], vi.fn(), onCitation)
    render(<Link href="#ref-c-1">1</Link>)
    fireEvent.click(screen.getByRole('button'))
    expect(onCitation).not.toHaveBeenCalled()
  })

  it('falls back to the source view when no citation handler is given', () => {
    const onRef = vi.fn()
    const Link = createCompactReferenceHoverComponent([cite({ source_id: 'doc-1' })], onRef)
    render(<Link href="#ref-c-1">1</Link>)
    fireEvent.click(screen.getByRole('button'))
    expect(onRef).toHaveBeenCalledWith('source', 'doc-1')
  })
})
