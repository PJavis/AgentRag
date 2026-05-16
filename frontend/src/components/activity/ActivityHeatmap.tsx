'use client'

import { useMemo } from 'react'
import type { ActivityHeatmapCell } from '@/lib/types/api'

const SCALE = [
  'bg-muted',
  'bg-emerald-200 dark:bg-emerald-900',
  'bg-emerald-400 dark:bg-emerald-700',
  'bg-emerald-600 dark:bg-emerald-500',
  'bg-emerald-800 dark:bg-emerald-300',
]

interface ActivityHeatmapProps {
  data: ActivityHeatmapCell[]
  days?: number
}

export function ActivityHeatmap({ data, days = 28 }: ActivityHeatmapProps) {
  const cells = useMemo(() => {
    const map: Record<string, number> = {}
    data.forEach((c) => {
      map[c.date] = c.count
    })
    const today = new Date()
    const out: { date: string; count: number; bucket: number }[] = []
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(today)
      d.setDate(today.getDate() - i)
      const key = d.toISOString().slice(0, 10)
      const count = map[key] ?? 0
      const bucket =
        count === 0 ? 0 : count < 3 ? 1 : count < 7 ? 2 : count < 15 ? 3 : 4
      out.push({ date: key, count, bucket })
    }
    return out
  }, [data, days])

  return (
    <div className="flex items-center gap-3">
      <div className="grid grid-rows-7 grid-flow-col gap-1">
        {cells.map((c) => (
          <div
            key={c.date}
            title={`${c.date}: ${c.count} events`}
            className={`h-3 w-3 rounded-sm ${SCALE[c.bucket]}`}
          />
        ))}
      </div>
      <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
        Less
        {SCALE.map((s, i) => (
          <span key={i} className={`h-3 w-3 rounded-sm ${s}`} />
        ))}
        More
      </div>
    </div>
  )
}

export default ActivityHeatmap
