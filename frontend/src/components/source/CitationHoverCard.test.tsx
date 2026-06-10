import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CitationHoverCard } from './CitationHoverCard'
import type { Citation } from '@/lib/types/api'

describe('CitationHoverCard', () => {
  it('renders trigger without crashing for a raptor summary citation', () => {
    render(
      <CitationHoverCard index={1} citation={{
        document_title: 'Tim mạch', node_level: 1, segment_type: 'raptor_summary',
        context_text: 'From the cardiology chapter', excerpt: 'summary text',
      }}>
        <button>ref</button>
      </CitationHoverCard>,
    )
    expect(screen.getByText('ref')).toBeInTheDocument()
  })
})
