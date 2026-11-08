import { describe, expect, it } from 'vitest'
import { chatFailure, dropClientFailures, mergeServerMessages, retryQuestionFor } from './chat-errors'

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

describe('mergeServerMessages', () => {
  const q = { id: 'srv-q', content: 'question' }
  const failed = { id: 'temp-ai-1', content: 'partial', error: 'stream error' }

  it('keeps a client-side failure that the server never persisted', () => {
    // Used to be wiped by setMessages(server) the tick after isSending flipped.
    const local = [{ id: 'temp-user-1', content: 'question' }, failed]
    expect(mergeServerMessages([q], local)).toEqual([q, failed])
  })

  it('returns the server copy untouched when nothing failed', () => {
    const server = [q, { id: 'srv-a', content: 'answer' }]
    expect(mergeServerMessages(server, [{ id: 'temp-user-1', content: 'question' }])).toBe(server)
  })

  it('does not duplicate a failure the server also has', () => {
    const persisted = { id: 'srv-a', content: 'x', error: 'boom' }
    expect(mergeServerMessages([q, persisted], [persisted])).toEqual([q, persisted])
  })
})

describe('dropClientFailures', () => {
  it('removes earlier failed bubbles when a new turn starts', () => {
    const msgs = [{ id: 'a', content: '' , error: 'x' }, { id: 'b', content: 'ok' }]
    expect(dropClientFailures(msgs)).toEqual([{ id: 'b', content: 'ok' }])
  })
})
