/**
 * Detecting failed chat turns.
 *
 * The backend persists an agent failure as a normal AI message
 * (`reasoning_path: 'error'`, answer `⚠️ Agent failed: <Exc>: <msg>`), and a
 * stream can also break mid-answer (client-side `error`). Both used to render
 * as ordinary answers with feedback/save buttons — or trigger a silent full
 * re-run. These helpers let the UI show an error state with Retry instead.
 */

const AGENT_FAILED_PREFIX = '⚠️ Agent failed:'

interface ChatLikeMessage {
  type: 'human' | 'ai'
  content: string
  reasoning_path?: string | null
  error?: string | null
}

export interface ChatFailure {
  failed: boolean
  /** Technical detail for the collapsed line under the friendly message. */
  detail: string
  /** Answer text produced before the failure (shown above the error). */
  partial: string
}

export function chatFailure(message: ChatLikeMessage): ChatFailure {
  if (message.type !== 'ai') return { failed: false, detail: '', partial: '' }
  if (message.error) {
    return { failed: true, detail: message.error, partial: message.content }
  }
  if (message.reasoning_path === 'error') {
    const content = message.content.trim()
    const detail = content.startsWith(AGENT_FAILED_PREFIX)
      ? content.slice(AGENT_FAILED_PREFIX.length).trim()
      : content
    return { failed: true, detail, partial: '' }
  }
  return { failed: false, detail: '', partial: '' }
}

/** The user question that produced the AI message at `index`, for Retry. */
export function retryQuestionFor(messages: ChatLikeMessage[], index: number): string | null {
  for (let i = index - 1; i >= 0; i--) {
    if (messages[i].type === 'human') return messages[i].content
  }
  return null
}

interface IdentifiedMessage {
  id: string
  error?: string | null
}

/**
 * Reconcile local messages with a fresh server copy without losing a
 * client-side failure. A turn that broke mid-stream is never persisted as an
 * assistant message, so a plain `setMessages(server)` (run whenever a send
 * finishes) erased the error bubble and its Retry one tick after it rendered.
 * The server copy already holds the user's question, so the failed bubble is
 * kept after it.
 */
export function mergeServerMessages<T extends IdentifiedMessage>(server: T[], local: T[]): T[] {
  const serverIds = new Set(server.map(m => m.id))
  const failures = local.filter(m => m.error && !serverIds.has(m.id))
  return failures.length ? [...server, ...failures] : server
}

/** A new turn supersedes an earlier failed one (it would sort after it otherwise). */
export function dropClientFailures<T extends IdentifiedMessage>(messages: T[]): T[] {
  return messages.some(m => m.error) ? messages.filter(m => !m.error) : messages
}
