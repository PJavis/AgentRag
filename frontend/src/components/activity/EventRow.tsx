'use client'

import { useRouter } from 'next/navigation'
import { Bot, FileUp, Search, CheckCircle, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'
import type { ActivityEvent } from '@/lib/types/api'

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  chat_turn: Bot,
  source_uploaded: FileUp,
  ingest_done: CheckCircle,
  ingest_failed: XCircle,
  search: Search,
}

interface EventRowProps {
  event: ActivityEvent
  onTrace?: (event: ActivityEvent) => void
}

export function EventRow({ event, onTrace }: EventRowProps) {
  const router = useRouter()
  const Icon = ICONS[event.event_type] ?? Bot
  const time = event.created_at
    ? new Date(event.created_at).toLocaleTimeString(undefined, { hour12: false })
    : ''
  const p = (event.payload || {}) as Record<string, unknown>

  const onClick = () => {
    if (event.event_type === 'chat_turn') {
      onTrace?.(event)
    } else if (
      event.event_type === 'source_uploaded' ||
      event.event_type === 'ingest_done' ||
      event.event_type === 'ingest_failed'
    ) {
      if (event.target_id) router.push(`/sources/${event.target_id}`)
    } else if (event.event_type === 'search') {
      toast.message(`Search: "${String(p.query ?? '')}"`)
    }
  }

  let summary = ''
  if (event.event_type === 'chat_turn') {
    summary = String(p.message ?? '').slice(0, 100)
  } else if (event.event_type === 'source_uploaded') {
    summary = String(p.filename ?? '')
    if (p.size_bytes) summary += ` · ${(Number(p.size_bytes) / 1024).toFixed(0)} KB`
  } else if (event.event_type === 'ingest_done') {
    summary = `${p.segment_count ?? 0} segments · ${p.duration_ms ?? 0} ms`
  } else if (event.event_type === 'ingest_failed') {
    summary = String(p.error ?? '').slice(0, 100)
  } else if (event.event_type === 'search') {
    summary = `"${p.query ?? ''}" · ${p.mode ?? ''} · ${p.hit_count ?? 0} hits`
  }

  return (
    <button
      onClick={onClick}
      className="flex w-full items-center gap-3 rounded border px-3 py-2 hover:bg-muted/50 text-left"
    >
      <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
      <span className="text-xs tabular-nums text-muted-foreground shrink-0 w-14">
        {time}
      </span>
      <Badge variant="outline" className="text-[10px] shrink-0">
        {event.event_type}
      </Badge>
      <span className="text-sm truncate">{summary}</span>
    </button>
  )
}

export default EventRow
