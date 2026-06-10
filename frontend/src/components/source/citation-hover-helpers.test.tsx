import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CitationContextLine, RaptorBadge } from './CitationHoverCard'

describe('citation hover helpers', () => {
  it('RaptorBadge shows level for node_level>=1', () => {
    render(<RaptorBadge nodeLevel={2} />)
    expect(screen.getByText(/L2/)).toBeInTheDocument()
  })
  it('RaptorBadge renders nothing for leaf nodes', () => {
    const { container } = render(<RaptorBadge nodeLevel={0} />)
    expect(container.textContent).toBe('')
  })
  it('CitationContextLine shows context text', () => {
    render(<CitationContextLine text="From cardiology" />)
    expect(screen.getByText(/From cardiology/)).toBeInTheDocument()
  })
  it('CitationContextLine renders nothing when empty', () => {
    const { container } = render(<CitationContextLine text={null} />)
    expect(container.textContent).toBe('')
  })
})
