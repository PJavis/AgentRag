'use client'

import { useState } from 'react'
import { Activity as ActivityIcon } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  useAdminActivityEvents,
  useAdminActivitySummary,
  useAdminUsers,
} from '@/lib/hooks/useActivity'
import { ActivityHeatmap } from '@/components/activity/ActivityHeatmap'
import { ActivityFeed } from '@/components/activity/ActivityFeed'
import { AppShell } from '@/components/layout/AppShell'

export default function AdminActivityPage() {
  const [userId, setUserId] = useState<string | undefined>(undefined)
  const [type, setType] = useState<string | undefined>(undefined)
  const users = useAdminUsers()
  const summary = useAdminActivitySummary(userId)
  const events = useAdminActivityEvents(userId, type)
  const s = summary.data

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="container mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <ActivityIcon className="h-5 w-5" />
          Admin · Activity
        </h1>
        <p className="text-sm text-muted-foreground">
          Global activity across all users. Requires admin role or X-Admin-Token.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">User:</span>
        <Select
          value={userId ?? 'all'}
          onValueChange={(v) => setUserId(v === 'all' ? undefined : v)}
        >
          <SelectTrigger className="w-72 h-8 text-xs">
            <SelectValue placeholder="All users" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all" className="text-xs">All users</SelectItem>
            {(users.data ?? []).map((u) => (
              <SelectItem
                key={u.user_id ?? 'anon'}
                value={u.user_id ?? 'anon'}
                className="text-xs"
              >
                {u.email ?? u.name ?? '(anonymous)'} · {u.event_count}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Activity heatmap (28d)</CardTitle>
        </CardHeader>
        <CardContent>
          <ActivityHeatmap data={s?.heatmap ?? []} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Events</CardTitle>
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
      </div>
    </AppShell>
  )
}
