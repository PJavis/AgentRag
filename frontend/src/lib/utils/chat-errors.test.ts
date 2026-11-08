import { describe, expect, it } from 'vitest'
import { chatFailure, retryQuestionFor } from './chat-errors'

describe('chatFailure', () => {
  it('flags a server-side agent failure persisted as an answer', () => {
    const f = chatFailure({
      type: 'ai',
      content: '⚠️ Agent failed: APIConnectionError: Connection refused',
      reasoning_path: 'error',
    })
    expect(f.failed).toBe(true)
    expect(f.detail).toBe('APIConnectionError: Connection refused')
    expect(f.partial).toBe('')
  })

  it('flags a stream that broke mid-answer and keeps the partial text', () => {
    const f = chatFailure({ type: 'ai', content: 'Nguyên nhân gồm', error: 'HTTP 502' })
    expect(f).toEqual({ failed: true, detail: 'HTTP 502', partial: 'Nguyên nhân gồm' })
  })

  it('does not flag a normal answer', () => {
    expect(chatFailure({ type: 'ai', content: 'Câu trả lời', reasoning_path: 'rag' }).failed).toBe(false)
  })

  it('never flags a human message', () => {
    expect(chatFailure({ type: 'human', content: 'x', error: 'boom' }).failed).toBe(false)
  })
})

describe('retryQuestionFor', () => {
  const messages = [
    { type: 'human' as const, content: 'câu 1' },
    { type: 'ai' as const, content: 'trả lời 1' },
    { type: 'human' as const, content: 'câu 2' },
    { type: 'ai' as const, content: '⚠️ Agent failed: X', reasoning_path: 'error' },
  ]

  it('returns the question right before the failed answer', () => {
    expect(retryQuestionFor(messages, 3)).toBe('câu 2')
  })

  it('returns null when there is no earlier question', () => {
    expect(retryQuestionFor([{ type: 'ai' as const, content: 'x' }], 0)).toBeNull()
  })
})
