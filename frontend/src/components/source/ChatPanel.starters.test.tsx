import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ChatPanel } from './ChatPanel'

vi.mock('@/lib/hooks/use-translation', () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}))
vi.mock('@/lib/hooks/use-modal-manager', () => ({
  useModalManager: () => ({ openModal: vi.fn() }),
}))
// Neutralize the source-references util: it transitively imports
// CitationHoverCard → @radix-ui/react-hover-card, which isn't installed and
// isn't needed for the empty-state (no AI messages rendered in these tests).
vi.mock('@/lib/utils/source-references', () => ({
  convertReferencesToCompactMarkdown: (s: string) => s,
  createCompactReferenceLinkComponent: () => () => null,
  createCompactReferenceHoverComponent: () => () => null,
}))

// jsdom doesn't implement scrollIntoView, which ChatPanel calls in an effect.
Element.prototype.scrollIntoView = vi.fn()

const baseProps = {
  messages: [],
  isStreaming: false,
  contextIndicators: null,
  onSendMessage: vi.fn(),
}

describe('ChatPanel starters', () => {
  it('renders fixed chips and dynamic starters', () => {
    render(<ChatPanel {...baseProps} dynamicStarters={['Ai là tác giả?']} />)
    expect(screen.getByText('📋 chat.starterSummary')).toBeInTheDocument()
    expect(screen.getByText('💡 Ai là tác giả?')).toBeInTheDocument()
  })

  it('shows skeletons while loading', () => {
    render(<ChatPanel {...baseProps} startersLoading />)
    expect(screen.getAllByTestId('starter-skeleton').length).toBe(3)
  })

  it('sends the prompt when a fixed chip is clicked', () => {
    const onSendMessage = vi.fn()
    render(<ChatPanel {...baseProps} onSendMessage={onSendMessage} />)
    fireEvent.click(screen.getByText('📋 chat.starterSummary'))
    expect(onSendMessage).toHaveBeenCalledWith(
      'Tóm tắt chi tiết tài liệu này', undefined, undefined, null,
    )
  })
})
