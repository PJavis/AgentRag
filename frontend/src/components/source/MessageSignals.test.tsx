import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MessageSignals, deriveCritiqueState } from './MessageSignals'
import type { NotebookChatMessage } from '@/lib/types/api'

vi.mock('@/lib/hooks/use-translation', () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}))

function msg(over: Partial<NotebookChatMessage>): NotebookChatMessage {
  return { id: '1', type: 'ai', content: 'a', ...over }
}

describe('deriveCritiqueState', () => {
  it('verified when critique ran and no corrective', () => {
    expect(deriveCritiqueState(msg({ timings_ms: { critique: 12 } }))).toBe('verified')
  })
  it('corrected when a corrective trace entry exists', () => {
    expect(deriveCritiqueState(msg({
      timings_ms: { critique: 12 },
      tool_trace: [{ tool_name: 'search_hybrid_kg', corrective: true }],
    }))).toBe('corrected')
  })
  it('null when critique never ran', () => {
    expect(deriveCritiqueState(msg({}))).toBeNull()
  })
})

describe('MessageSignals', () => {
  it('shows fast-path + cache chips', () => {
    render(<MessageSignals message={msg({ reasoning_path: 'fast', semantic_cache_hit: true })} />)
    expect(screen.getByText('ragSignals.fastPath')).toBeInTheDocument()
    expect(screen.getByText('ragSignals.instantCached')).toBeInTheDocument()
  })
  it('renders nothing when no signals present', () => {
    const { container } = render(<MessageSignals message={msg({})} />)
    expect(container.textContent).toBe('')
  })
})
