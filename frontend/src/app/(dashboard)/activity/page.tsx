'use client'

import { useState } from 'react'
import { Activity as ActivityIcon, Bot, FileUp, Search, DollarSign } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  useActivitySummary,
  useActivityEvents,
} from '@/lib/hooks/useActivity'
import { ActivityHeatmap } from '@/components/activity/ActivityHeatmap'
import { ActivityFeed } from '@/components/activity/ActivityFeed'

function fmtInt(n: number): string {
  return n.toLocaleString()
}

function fmtUsd(n: number): string {
  if (!n) return '$0'
  if (n < 1) return `$${n.toFixed(4)}`
  return `$${n.toFixed(2)}`
}

function SummaryCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
}) {
  return (
    <Card>
      <CardContent className="p-4 flex items-center gap-3">
        <Icon className="h-5 w-5 text-muted-foreground" />
        <div>
          <div className="text-xs text-muted-foreground">{label}</div>
          <div className="text-2xl font-semibold tabular-nums">{value}</div>
        </div>
      </CardContent>
    </Card>
  )
}

export default function ActivityPage() {
  const [type, setType] = useState<string | undefined>(undefined)
  const summary = useActivitySummary()
  const events = useActivityEvents(type, 50)
  const s = summary.data

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <ActivityIcon className="h-5 w-5" />
          Activity
        </h1>
        <p className="text-sm text-muted-foreground">
          Your recent chats, uploads, and searches. Auto-refresh every 30s.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <SummaryCard icon={Bot} label="Chats" value={fmtInt(s?.counts.chat_turn ?? 0)} />
        <SummaryCard icon={FileUp} label="Uploads" value={fmtInt(s?.counts.source_uploaded ?? 0)} />
        <SummaryCard icon={Search} label="Searches" value={fmtInt(s?.counts.search ?? 0)} />
        <SummaryCard icon={DollarSign} label="Est. cost" value={fmtUsd(s?.usd_estimate ?? 0)} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Last 28 days</CardTitle>
        </CardHeader>
        <CardContent>
          <ActivityHeatmap data={s?.heatmap ?? []} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Recent events</CardTitle>
        </CardHeader>
        <CardContent>
          <ActivityFeed
            events={events.data?.entries ?? []}
            loading={events.isLoading}
            onTypeChange={setType}
          />
        </CardContent>
      </Card>
    </div>
  )
}
