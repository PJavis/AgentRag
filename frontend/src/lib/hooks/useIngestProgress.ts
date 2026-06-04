'use client'

import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { getApiUrl } from '@/lib/config'
import { useAuthStore } from '@/lib/stores/auth-store'

/**
 * Subscribe to the backend ingest-progress SSE stream and refresh the source
 * lists live on each stage transition (queued→parsing→searchable→enriching→
 * done). Uses a fetch-based reader (not EventSource) so the Bearer token can be
 * sent. Reconnects with backoff if the stream drops. Mount once (AppShell).
 */
export function useIngestProgress(enabled: boolean = true) {
  const queryClient = useQueryClient()

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    let controller: AbortController | null = null
    let retryMs = 1000

    const refresh = () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] })
    }

    async function connect() {
      while (!cancelled) {
        controller = new AbortController()
        try {
          const apiUrl = await getApiUrl()
          const token = useAuthStore.getState().token
          if (!token) {
            await new Promise((r) => setTimeout(r, 3000))
            continue
          }
          const resp = await fetch(`${apiUrl}/api/sources/progress/stream`, {
            headers: { Authorization: `Bearer ${token}` },
            signal: controller.signal,
          })
          if (!resp.ok || !resp.body) throw new Error(`stream ${resp.status}`)
          retryMs = 1000 // reset backoff on successful connect
          const reader = resp.body.getReader()
          const decoder = new TextDecoder()
          let buf = ''
          while (!cancelled) {
            const { done, value } = await reader.read()
            if (done) break
            buf += decoder.decode(value, { stream: true })
            const blocks = buf.split('\n\n')
            buf = blocks.pop() || ''
            for (const block of blocks) {
              const dataLine = block.split('\n').find((l) => l.startsWith('data:'))
              if (!dataLine) continue
              try {
                JSON.parse(dataLine.slice(5).trim()) // validate; payload unused
                refresh()
              } catch {
                /* ignore malformed frame */
              }
            }
          }
        } catch {
          /* network / abort — fall through to backoff */
        }
        if (cancelled) break
        await new Promise((r) => setTimeout(r, retryMs))
        retryMs = Math.min(retryMs * 2, 15000)
      }
    }

    connect()
    return () => {
      cancelled = true
      controller?.abort()
    }
  }, [enabled, queryClient])
}
