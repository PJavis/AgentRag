'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { getApiErrorMessage } from '@/lib/utils/error-handler'
import { useTranslation } from '@/lib/hooks/use-translation'
import { chatApi } from '@/lib/api/chat'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { getApiUrl } from '@/lib/config'
import { useAuthStore } from '@/lib/stores/auth-store'
import {
  NotebookChatMessage,
  CreateNotebookChatSessionRequest,
  UpdateNotebookChatSessionRequest,
  SourceListResponse,
  NoteResponse,
  DomainFilterValue
} from '@/lib/types/api'
import { ContextSelections } from '@/app/(dashboard)/notebooks/[id]/page'

// Live streaming step surfaced to the UI flow indicator. Mirrors the backend
// `event: status` step values plus a few client-side phases.
export type StreamingStep =
  | 'idle'
  | 'connecting'
  | 'chitchat'
  | 'classify'
  | 'structured_reasoning'
  | 'retrieve'
  | 'decide'
  | 'tool'
  | 'answer'
  | 'writing'
  | 'finalizing'

interface UseNotebookChatParams {
  notebookId: string
  sources: SourceListResponse[]
  notes: NoteResponse[]
  contextSelections: ContextSelections
}

export function useNotebookChat({ notebookId, sources, notes, contextSelections }: UseNotebookChatParams) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<NotebookChatMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [tokenCount, setTokenCount] = useState<number>(0)
  const [charCount, setCharCount] = useState<number>(0)
  // Pending model override for when user changes model before a session exists
  const [pendingModelOverride, setPendingModelOverride] = useState<string | null>(null)
  // Streaming progress for richer "flow" UI feedback. The backend emits
  // `event: status` frames with a `step` field as the agent moves through its
  // pipeline; we surface that live so the user sees what's happening instead of
  // a dumb spinner. Steps (from src/agentrag/agent/service.py):
  //   connecting          → client-side: request sent, no HTTP response yet
  //   chitchat            → fast-path small-talk reply
  //   classify            → intent classification
  //   structured_reasoning→ structured (SQL/table) pipeline running
  //   retrieve            → semantic/KG retrieval
  //   decide              → agent deciding next tool
  //   tool                → executing a tool (carries `tool` name)
  //   answer              → composing the final answer (tokens follow)
  //   writing             → client-side: first token arrived
  //   finalizing          → client-side: stream ended, reconciling server state
  //   idle                → no stream in flight
  const [streamingStep, setStreamingStep] = useState<StreamingStep>('idle')
  const [streamingTool, setStreamingTool] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Fetch sessions for this notebook
  const {
    data: sessions = [],
    isLoading: loadingSessions,
    refetch: refetchSessions
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSessions(notebookId),
    queryFn: () => chatApi.listSessions(notebookId),
    enabled: !!notebookId
  })

  // Fetch current session with messages
  const {
    data: currentSession,
    refetch: refetchCurrentSession
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!),
    queryFn: () => chatApi.getSession(currentSessionId!),
    enabled: !!notebookId && !!currentSessionId
  })

  // Sync local messages with server state on session changes / refetches.
  // Guard against mid-send: while isSending=true, TanStack may surface a
  // stale `currentSession.messages` (cache race) and wipe the optimistic
  // user bubble. Skip until the send finishes — sendMessage handles the
  // final reconciliation itself.
  useEffect(() => {
    if (!currentSession?.messages) return
    if (isSending) return
    setMessages(currentSession.messages)
  }, [currentSession, isSending])

  // Auto-select most recent session when sessions are loaded
  useEffect(() => {
    if (sessions.length > 0 && !currentSessionId) {
      // Sessions are sorted by created date desc from API
      const mostRecentSession = sessions[0]
      setCurrentSessionId(mostRecentSession.id)
    }
  }, [sessions, currentSessionId])

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (data: CreateNotebookChatSessionRequest) =>
      chatApi.createSession(data),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      setCurrentSessionId(newSession.id)
      toast.success(t('chat.sessionCreated'))
    },
    onError: (err: unknown) => {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToCreateSession'))
    }
  })

  // Update session mutation
  const updateSessionMutation = useMutation({
    mutationFn: ({ sessionId, data }: {
      sessionId: string
      data: UpdateNotebookChatSessionRequest
    }) => chatApi.updateSession(sessionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!)
      })
      toast.success(t('chat.sessionUpdated'))
    },
    onError: (err: unknown) => {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToUpdateSession'))
    }
  })

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) =>
      chatApi.deleteSession(sessionId),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      if (currentSessionId === deletedId) {
        setCurrentSessionId(null)
        setMessages([])
      }
      toast.success(t('chat.sessionDeleted'))
    },
    onError: (err: unknown) => {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToDeleteSession'))
    }
  })

  // Build context from sources and notes based on user selections
  const buildContext = useCallback(async () => {
    // Build context_config mapping IDs to selection modes
    const context_config: { sources: Record<string, string>, notes: Record<string, string> } = {
      sources: {},
      notes: {}
    }

    // Map source selections
    sources.forEach(source => {
      const mode = contextSelections.sources[source.id]
      if (mode === 'insights') {
        context_config.sources[source.id] = 'insights'
      } else if (mode === 'full') {
        context_config.sources[source.id] = 'full content'
      } else {
        context_config.sources[source.id] = 'not in'
      }
    })

    // Map note selections
    notes.forEach(note => {
      const mode = contextSelections.notes[note.id]
      if (mode === 'full') {
        context_config.notes[note.id] = 'full content'
      } else {
        context_config.notes[note.id] = 'not in'
      }
    })

    // Call API to build context with actual content
    const response = await chatApi.buildContext({
      notebook_id: notebookId,
      context_config
    })

    // Store token and char counts
    setTokenCount(response.token_count)
    setCharCount(response.char_count)

    return response.context
  }, [notebookId, sources, notes, contextSelections])

  // Send message (synchronous, no streaming)
  const sendMessage = useCallback(async (
    message: string,
    modelOverride?: string,
    domainFilter?: DomainFilterValue | null,
    verbosity?: 'concise' | 'detailed' | null
  ) => {
    let sessionId = currentSessionId

    // Auto-create session if none exists
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30
          ? `${message.substring(0, 30)}...`
          : message
        const newSession = await chatApi.createSession({
          notebook_id: notebookId,
          title: defaultTitle,
          // Include pending model override when creating session
          model_override: pendingModelOverride ?? undefined
        })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        // Clear pending model override now that it's applied to the session
        setPendingModelOverride(null)
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
        })
      } catch (err: unknown) {
        const error = err as { response?: { data?: { detail?: string } }, message?: string };
        toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToCreateSession'))
        return
      }
    }

    // Add user message optimistically
    const userMessage: NotebookChatMessage = {
      id: `temp-${Date.now()}`,
      type: 'human',
      content: message,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMessage])
    setIsSending(true)

    try {
      // Build context and send message
      const context = await buildContext()
      // Strip out empty fields so backend treats unset = no filter
      const hasSystem = !!domainFilter?.system
      const hasSpecialties = Array.isArray(domainFilter?.specialties) && (domainFilter?.specialties?.length ?? 0) > 0
      const normalizedDomainFilter: DomainFilterValue | undefined =
        hasSystem || hasSpecialties
          ? {
              ...(hasSystem ? { system: domainFilter!.system } : {}),
              ...(hasSpecialties ? { specialties: domainFilter!.specialties } : {})
            }
          : undefined

      const response = await chatApi.sendMessage({
        session_id: sessionId,
        message,
        context,
        model_override: modelOverride ?? (currentSession?.model_override ?? undefined),
        domain_filter: normalizedDomainFilter,
        verbosity: verbosity ?? undefined
      })

      // Only replace local messages when server returns the expected shape.
      // If server returns fewer messages than we already have locally (e.g.
      // stale list_messages cache, transient race), don't wipe the user
      // bubble — the next refetch will fix it.
      const serverMessages = response?.messages
      if (Array.isArray(serverMessages) && serverMessages.length >= 2) {
        setMessages(serverMessages)
      }

      // Refetch current session to get updated data
      await refetchCurrentSession()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      console.error('Error sending message:', error)
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToSendMessage'))
      // Keep the optimistic user bubble visible. Backend persists the user
      // turn before agent.chat() runs, so refetch will reconcile when the
      // session DB write lands.
      try {
        await refetchCurrentSession()
      } catch {
        // refetch fail → at least keep the optimistic user turn visible.
      }
    } finally {
      setIsSending(false)
    }
  }, [
    notebookId,
    currentSessionId,
    currentSession,
    pendingModelOverride,
    buildContext,
    refetchCurrentSession,
    queryClient,
    t
  ]) // eslint-disable-line react-hooks/exhaustive-deps

  // Send a message with an attached image — routes to /chat/with-image
  // (ad-hoc multimodal Q&A, no retrieval, vision LLM grounds on the bytes).
  const sendImageMessage = useCallback(async (
    message: string,
    file: File,
  ) => {
    let sessionId = currentSessionId
    if (!sessionId) {
      try {
        const newSession = await chatApi.createSession({
          notebook_id: notebookId,
          title: (message || file.name).slice(0, 30),
          model_override: pendingModelOverride ?? undefined,
        })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        setPendingModelOverride(null)
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSessions(notebookId) })
      } catch (err: unknown) {
        const e = err as { response?: { data?: { detail?: string } }, message?: string }
        toast.error(getApiErrorMessage(e.response?.data?.detail || e.message, (k) => t(k), 'apiErrors.failedToCreateSession'))
        return
      }
    }

    // Optimistic user bubble showing the local image preview.
    const localUrl = URL.createObjectURL(file)
    const tempId = `temp-user-img-${Date.now()}`
    setMessages(prev => [
      ...prev,
      {
        id: tempId,
        type: 'human',
        content: message || 'Mô tả hình này',
        timestamp: new Date().toISOString(),
        // attached image preview rendered by ChatPanel via extra_metadata
        // (server message will replace this with persisted URL).
        attached_image_url: localUrl,
      } as NotebookChatMessage & { attached_image_url?: string },
    ])
    setIsSending(true)

    try {
      const response = await chatApi.sendWithImage(sessionId, message, file)
      if (Array.isArray(response?.messages) && response.messages.length >= 2) {
        setMessages(response.messages)
      }
      await refetchCurrentSession()
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } }, message?: string }
      console.error('sendImageMessage error', e)
      toast.error(getApiErrorMessage(e.response?.data?.detail || e.message, (k) => t(k), 'apiErrors.failedToSendMessage'))
      try { await refetchCurrentSession() } catch {}
    } finally {
      URL.revokeObjectURL(localUrl)
      setIsSending(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, notebookId, pendingModelOverride, queryClient, t, refetchCurrentSession])

  // Streaming variant of sendMessage — consumes /chat/execute-stream SSE,
  // appends tokens to a placeholder AI bubble for live "typing" feel. Falls
  // back to non-streaming on any error.
  const sendMessageStreaming = useCallback(async (
    message: string,
    modelOverride?: string,
    domainFilter?: DomainFilterValue | null,
    verbosity?: 'concise' | 'detailed' | null
  ) => {
    let sessionId = currentSessionId
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30 ? `${message.substring(0, 30)}...` : message
        const newSession = await chatApi.createSession({
          notebook_id: notebookId,
          title: defaultTitle,
          model_override: pendingModelOverride ?? undefined
        })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        setPendingModelOverride(null)
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSessions(notebookId) })
      } catch (err: unknown) {
        const error = err as { response?: { data?: { detail?: string } }, message?: string }
        toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToCreateSession'))
        return
      }
    }

    // Optimistic user bubble + empty placeholder AI bubble we'll fill in.
    const userId = `temp-user-${Date.now()}`
    const aiPlaceholderId = `temp-ai-${Date.now()}`
    setMessages(prev => [
      ...prev,
      { id: userId, type: 'human', content: message, timestamp: new Date().toISOString() },
      { id: aiPlaceholderId, type: 'ai', content: '', timestamp: new Date().toISOString() },
    ])
    setIsSending(true)
    setStreamingStep('connecting')
    setStreamingTool(null)
    // Fresh AbortController for this stream so cancel() can interrupt it.
    abortControllerRef.current?.abort()
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    const hasSystem = !!domainFilter?.system
    const hasSpecialties = Array.isArray(domainFilter?.specialties) && (domainFilter?.specialties?.length ?? 0) > 0
    const normalizedDomainFilter: DomainFilterValue | undefined =
      hasSystem || hasSpecialties
        ? {
            ...(hasSystem ? { system: domainFilter!.system } : {}),
            ...(hasSpecialties ? { specialties: domainFilter!.specialties } : {})
          }
        : undefined

    try {
      const apiUrl = await getApiUrl()
      const token = useAuthStore.getState().token
      const resp = await fetch(`${apiUrl}/api/chat/execute-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          session_id: sessionId,
          message,
          context: { sources: [], notes: [] },
          // Ticked sources → scope retrieval to just those documents (speed + focus).
          source_ids: sources
            .filter(s => {
              const m = contextSelections.sources[s.id]
              return m === 'insights' || m === 'full'
            })
            .map(s => s.id),
          model_override: modelOverride ?? (currentSession?.model_override ?? undefined),
          domain_filter: normalizedDomainFilter,
          verbosity: verbosity ?? undefined,
        }),
        signal: abortController.signal,
      })
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let accumulated = ''
      let reasoningPath = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        // SSE frames are separated by blank line ("\n\n").
        let sep
        while ((sep = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, sep)
          buffer = buffer.slice(sep + 2)
          let eventName = 'message'
          let dataLine = ''
          for (const line of frame.split('\n')) {
            if (line.startsWith('event: ')) eventName = line.slice(7).trim()
            else if (line.startsWith('data: ')) dataLine = line.slice(6)
          }
          if (!dataLine) continue
          try {
            const data = JSON.parse(dataLine)
            if (eventName === 'status' && typeof data.step === 'string') {
              // Live agent pipeline step → drive the flow indicator.
              setStreamingStep(data.step as StreamingStep)
              setStreamingTool(typeof data.tool === 'string' ? data.tool : null)
            } else if (eventName === 'token' && typeof data.text === 'string') {
              accumulated += data.text
              if (accumulated.length > 0) setStreamingStep('writing')
              setMessages(prev =>
                prev.map(m => (m.id === aiPlaceholderId ? { ...m, content: accumulated } : m))
              )
            } else if (eventName === 'done') {
              reasoningPath = String(data.reasoning_path || '')
            } else if (eventName === 'error') {
              // Backend emits {"message": ...}; older paths use {"detail": ...}.
              throw new Error(String(data.message || data.detail || 'stream error'))
            }
          } catch (frameErr) {
            // Re-throw stream `error` events; swallow only JSON-parse noise.
            if (eventName === 'error') throw frameErr
          }
        }
      }
      setStreamingStep('finalizing')
      // Final reconciliation — refetch real session to pick up server-side
      // message_ids, citations, follow_ups, etc.
      if (reasoningPath) {
        setMessages(prev =>
          prev.map(m => (m.id === aiPlaceholderId ? { ...(m as object), reasoning_path: reasoningPath } as NotebookChatMessage : m))
        )
      }
      await refetchCurrentSession()
    } catch (err: unknown) {
      const aborted = (err as { name?: string })?.name === 'AbortError' || abortController.signal.aborted
      if (aborted) {
        // User cancelled. Keep any partial assistant content already streamed
        // (gives them what was produced so far); drop empty placeholder.
        setMessages(prev =>
          prev.filter(m => !(m.id === aiPlaceholderId && !m.content))
        )
        toast.info('Generation stopped')
      } else {
        console.warn('streaming failed, falling back to non-streaming:', err)
        // Drop the placeholder bubbles and fall back to the existing path.
        setMessages(prev => prev.filter(m => m.id !== userId && m.id !== aiPlaceholderId))
        await sendMessage(message, modelOverride, domainFilter, verbosity)
      }
    } finally {
      setIsSending(false)
      setStreamingStep('idle')
      setStreamingTool(null)
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, currentSession, notebookId, pendingModelOverride, queryClient, t, refetchCurrentSession])

  // Regenerate a specific assistant message
  const regenerateMessage = useCallback(async (
    assistantMessageId: string,
    domainFilter?: DomainFilterValue | null,
    verbosity?: 'concise' | 'detailed' | null
  ) => {
    if (!currentSessionId) return
    const hasSystem = !!domainFilter?.system
    const hasSpecialties = Array.isArray(domainFilter?.specialties) && (domainFilter?.specialties?.length ?? 0) > 0
    const normalizedDomainFilter: DomainFilterValue | undefined =
      hasSystem || hasSpecialties
        ? {
            ...(hasSystem ? { system: domainFilter!.system } : {}),
            ...(hasSpecialties ? { specialties: domainFilter!.specialties } : {})
          }
        : undefined

    setIsSending(true)
    // Optimistic: drop the stale assistant bubble AND every turn after it.
    // Regenerating invalidates that downstream conversation branch — the
    // server will purge them too, but pre-emptively hide them so the user
    // does not see ghost turns during the round-trip.
    setMessages(prev => {
      const idx = prev.findIndex(m => m.id === assistantMessageId)
      return idx < 0 ? prev : prev.slice(0, idx)
    })
    try {
      const response = await chatApi.regenerate({
        session_id: currentSessionId,
        assistant_message_id: assistantMessageId,
        domain_filter: normalizedDomainFilter,
        verbosity: verbosity ?? undefined,
        model_override: currentSession?.model_override ?? undefined
      })
      const serverMessages = response?.messages
      if (Array.isArray(serverMessages) && serverMessages.length >= 2) {
        setMessages(serverMessages)
      }
      await refetchCurrentSession()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      console.error('Error regenerating message:', error)
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToSendMessage'))
      try { await refetchCurrentSession() } catch {}
    } finally {
      setIsSending(false)
    }
  }, [currentSessionId, currentSession, refetchCurrentSession, t]) // eslint-disable-line react-hooks/exhaustive-deps

  // Switch session
  const switchSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
  }, [])

  // Cancel in-flight streaming. Triggers AbortError in fetch → caller's finally
  // clears state and keeps any partial assistant content already received.
  const cancelStreaming = useCallback(() => {
    abortControllerRef.current?.abort()
  }, [])

  // Create session
  const createSession = useCallback((title?: string) => {
    return createSessionMutation.mutate({
      notebook_id: notebookId,
      title
    })
  }, [createSessionMutation, notebookId])

  // Update session
  const updateSession = useCallback((sessionId: string, data: UpdateNotebookChatSessionRequest) => {
    return updateSessionMutation.mutate({
      sessionId,
      data
    })
  }, [updateSessionMutation])

  // Delete session
  const deleteSession = useCallback((sessionId: string) => {
    return deleteSessionMutation.mutate(sessionId)
  }, [deleteSessionMutation])

  // Set model override - handles both existing sessions and pending state
  const setModelOverride = useCallback((model: string | null) => {
    if (currentSessionId) {
      // Session exists - update it directly
      updateSessionMutation.mutate({
        sessionId: currentSessionId,
        data: { model_override: model }
      })
    } else {
      // No session yet - store as pending
      setPendingModelOverride(model)
    }
  }, [currentSessionId, updateSessionMutation])

  // Update token/char counts when context selections change
  useEffect(() => {
    const updateContextCounts = async () => {
      try {
        await buildContext()
      } catch (error) {
        console.error('Error updating context counts:', error)
      }
    }
    updateContextCounts()
  }, [buildContext])

  return {
    // State
    sessions,
    currentSession: currentSession || sessions.find(s => s.id === currentSessionId),
    currentSessionId,
    messages,
    isSending,
    streamingStep,
    streamingTool,
    loadingSessions,
    tokenCount,
    charCount,
    pendingModelOverride,

    // Actions
    createSession,
    updateSession,
    deleteSession,
    switchSession,
    sendMessage,
    sendMessageStreaming,
    sendImageMessage,
    regenerateMessage,
    cancelStreaming,
    setModelOverride,
    refetchSessions
  }
}
