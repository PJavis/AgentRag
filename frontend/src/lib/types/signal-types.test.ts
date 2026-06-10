import { describe, it, expect } from 'vitest'
import type { Citation, NotebookChatMessage } from './api'

describe('RAG signal type fields', () => {
  it('Citation accepts node_level, context_text, raptor_summary segment_type', () => {
    const c: Citation = {
      document_title: 'x', node_level: 1, context_text: 'ctx',
      segment_type: 'raptor_summary',
    }
    expect(c.node_level).toBe(1)
    expect(c.segment_type).toBe('raptor_summary')
  })
  it('NotebookChatMessage accepts semantic_cache_hit + domain_route + retrieval_mode', () => {
    const m: NotebookChatMessage = {
      id: '1', type: 'ai', content: 'a',
      semantic_cache_hit: true, domain_route: 'tim_mach', retrieval_mode: 'hybrid_kg',
    }
    expect(m.semantic_cache_hit).toBe(true)
  })
})
