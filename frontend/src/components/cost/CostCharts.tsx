'use client'

import {
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip as RTooltip,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { CostBucket, CostEntry } from '@/lib/api/metrics'
import { perModelSpend, costOverTime } from '@/lib/utils/cost-charts'

const AXIS = { fontSize: 10 }

export function CostCharts({
  perModel, recent,
}: { perModel: Record<string, CostBucket>; recent: CostEntry[] }) {
  const spend = perModelSpend(perModel)
  const series = costOverTime(recent)

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader><CardTitle className="text-sm">Cost over time</CardTitle></CardHeader>
        <CardContent className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="label" tick={AXIS} />
              <YAxis tick={AXIS} />
              <RTooltip />
              <Area type="monotone" dataKey="usd" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.2} />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-sm">Spend by model</CardTitle></CardHeader>
        <CardContent className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={spend}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={AXIS} />
              <YAxis tick={AXIS} />
              <RTooltip />
              <Bar dataKey="usd" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}

export default CostCharts
