'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { EventRow } from './EventRow'
import type { ActivityEvent } from '@/lib/types/api'

const TYPES: { value: string | undefined; label: string }[] = [
  { value: undefined, label: 'All' },
  { value: 'chat_turn', label: 'Chats' },
  { value: 'source_uploaded', label: 'Uploads' },
  { value: 'search', label: 'Searches' },
]

interface ActivityFeedProps {
  events: ActivityEvent[]
  loading?: boolean
  onTypeChange?: (t: string | undefined) => void
  onTrace?: (event: ActivityEvent) => void
}

export function ActivityFeed({
  events,
  loading,
  onTypeChange,
  onTrace,
}: ActivityFeedProps) {
  const [active, setActive] = useState<string | undefined>(undefined)
  useEffect(() => {
    onTypeChange?.(active)
  }, [active, onTypeChange])

  return (
    <div className="space-y-2">
      <div className="flex gap-1.5">
        {TYPES.map((t) => (
          <Button
            key={t.label}
            size="sm"
            variant={active === t.value ? 'default' : 'outline'}
            className="h-7 text-xs"
            onClick={() => setActive(t.value)}
          >
            {t.label}
          </Button>
        ))}
      </div>
      {loading ? (
        <div className="text-center text-muted-foreground py-6 text-sm">
          Loading…
        </div>
      ) : events.length === 0 ? (
        <div className="text-center text-muted-foreground py-6 text-sm">
          No activity yet
        </div>
      ) : (
        <div className="space-y-1.5">
          {events.map((e) => (
            <EventRow key={e.id} event={e} onTrace={onTrace} />
          ))}
        </div>
      )}
    </div>
  )
}

export default ActivityFeed
