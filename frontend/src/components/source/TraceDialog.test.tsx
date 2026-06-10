import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TraceDialog, groupMultiHop } from './TraceDialog'
import type { NotebookChatMessage } from '@/lib/types/api'

describe('groupMultiHop', () => {
  it('flags multihop and corrective entries', () => {
    const grouped = groupMultiHop([
      { tool_name: 'a', multihop: true },
      { tool_name: 'b', corrective: true },
      { tool_name: 'c' },
    ])
    expect(grouped[0].isMultiHop).toBe(true)
    expect(grouped[1].isCorrective).toBe(true)
    expect(grouped[2].isMultiHop).toBe(false)
  })
})

describe('TraceDialog', () => {
  const base: NotebookChatMessage = {
    id: '1', type: 'ai', content: 'a',
    timings_ms: { total: 100, answer: 50 },
  }
  it('shows fast-path explainer when reasoning_path=fast', () => {
    render(<TraceDialog open onOpenChange={() => {}}
      message={{ ...base, reasoning_path: 'fast' }} />)
    expect(screen.getByText(/fast path/i)).toBeInTheDocument()
  })
  it('labels a corrective tool entry', () => {
    render(<TraceDialog open onOpenChange={() => {}}
      message={{ ...base, tool_trace: [{ tool_name: 'search_hybrid_kg', corrective: true }] }} />)
    expect(screen.getByText(/corrective/i)).toBeInTheDocument()
  })
})
