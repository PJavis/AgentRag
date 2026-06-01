'use client'

import { useQuery } from '@tanstack/react-query'
import { sourceChatApi } from '@/lib/api/source-chat'
import { chatApi } from '@/lib/api/chat'

interface UseChatStartersArgs {
  kind: 'source' | 'notebook'
  id: string
  enabled: boolean
}

/**
 * Fetch document-aware starter questions for the empty chat state.
 * Only runs while `enabled` (caller passes true when messages.length === 0).
 * Errors are swallowed → starters: [] so the UI falls back to fixed chips.
 */
export function useChatStarters({ kind, id, enabled }: UseChatStartersArgs) {
  const query = useQuery({
    queryKey: ['chatStarters', kind, id],
    queryFn: () =>
      kind === 'source' ? sourceChatApi.getStarters(id) : chatApi.getStarters(id),
    enabled: enabled && !!id,
    staleTime: 10 * 60 * 1000,
    retry: false,
  })
  return {
    starters: query.data ?? [],
    isLoading: query.isLoading && query.fetchStatus !== 'idle',
  }
}
