'use client'

import { useState } from 'react'
import { RotateCcw, Activity, DollarSign, Cpu, Clock } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { AppShell } from '@/components/layout/AppShell'
import { CostCharts } from '@/components/cost/CostCharts'
import {
  useCostSummary,
  useRecentCostCalls,
  useResetCostLedger,
} from '@/lib/hooks/useCostMetrics'
import type { CostBucket } from '@/lib/api/metrics'

function fmtUsd(n: number): string {
  if (n < 0.0001) return '$0'
  if (n < 1) return `$${n.toFixed(4)}`
  return `$${n.toFixed(2)}`
}

function fmtInt(n: number): string {
  return n.toLocaleString()
}

function fmtTs(epochSec: number): string {
  if (!epochSec) return '—'
  const d = new Date(epochSec * 1000)
  return d.toLocaleTimeString(undefined, { hour12: false })
}

function SummaryCard({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  hint?: string
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <Icon className="h-5 w-5 text-muted-foreground" />
          <div>
            <div className="text-xs text-muted-foreground">{label}</div>
            <div className="text-2xl font-semibold tabular-nums">{value}</div>
            {hint && <div className="text-[10px] text-muted-foreground mt-0.5">{hint}</div>}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function BucketTable({
  title,
  buckets,
}: {
  title: string
  buckets: Record<string, CostBucket>
}) {
  const rows = Object.entries(buckets)
    .map(([key, b]) => ({ key, ...b }))
    .sort((a, b) => b.usd - a.usd)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/40 text-xs uppercase">
              <tr>
                <th className="text-left px-4 py-2 font-medium">Key</th>
                <th className="text-right px-4 py-2 font-medium">Calls</th>
                <th className="text-right px-4 py-2 font-medium">In tok</th>
                <th className="text-right px-4 py-2 font-medium">Out tok</th>
                <th className="text-right px-4 py-2 font-medium">Avg ms</th>
                <th className="text-right px-4 py-2 font-medium">p50</th>
                <th className="text-right px-4 py-2 font-medium">p95</th>
                <th className="text-right px-4 py-2 font-medium">USD</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={8} className="text-center text-muted-foreground py-6">
                    No data
                  </td>
                </tr>
              ) : (
                rows.map((r) => (
                  <tr key={r.key} className="border-t">
                    <td className="px-4 py-2 font-mono text-xs">{r.key}</td>
                    <td className="text-right tabular-nums px-4 py-2">{r.calls}</td>
                    <td className="text-right tabular-nums px-4 py-2">{fmtInt(r.in_tokens)}</td>
                    <td className="text-right tabular-nums px-4 py-2">{fmtInt(r.out_tokens)}</td>
                    <td className="text-right tabular-nums px-4 py-2">{r.avg_latency_ms}</td>
                    <td className="text-right tabular-nums px-4 py-2 text-muted-foreground">{r.p50_latency_ms ?? '—'}</td>
                    <td className="text-right tabular-nums px-4 py-2 text-muted-foreground">{r.p95_latency_ms ?? '—'}</td>
                    <td className="text-right tabular-nums px-4 py-2 font-medium">{fmtUsd(r.usd)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}

export default function CostDashboardPage() {
  const [recentLimit, setRecentLimit] = useState(50)
  const summary = useCostSummary()
  const recent = useRecentCostCalls(recentLimit)
  const reset = useResetCostLedger()

  const s = summary.data
  const r = recent.data ?? []

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">LLM Cost & Token Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            In-memory ledger — cleared on server restart. Auto-refresh every 5s.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => reset.mutate()}
          disabled={reset.isPending}
          className="gap-2"
        >
          <RotateCcw className="h-4 w-4" />
          Reset
        </Button>
      </div>

      {summary.isLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <SummaryCard
            icon={Activity}
            label="Total calls"
            value={fmtInt(s?.total_calls ?? 0)}
          />
          <SummaryCard
            icon={Cpu}
            label="Input tokens"
            value={fmtInt(s?.total_in_tokens ?? 0)}
          />
          <SummaryCard
            icon={Cpu}
            label="Output tokens"
            value={fmtInt(s?.total_out_tokens ?? 0)}
          />
          <SummaryCard
            icon={DollarSign}
            label="Estimated cost"
            value={fmtUsd(s?.total_usd ?? 0)}
            hint={s?.note ? 'USD estimate' : undefined}
          />
        </div>
      )}

      <Tabs defaultValue="task" className="w-full">
        <TabsList>
          <TabsTrigger value="task">By task</TabsTrigger>
          <TabsTrigger value="model">By model</TabsTrigger>
          <TabsTrigger value="recent">Recent calls</TabsTrigger>
          <TabsTrigger value="charts">Charts</TabsTrigger>
        </TabsList>

        <TabsContent value="task" className="mt-3">
          <BucketTable title="Per-task cost" buckets={s?.per_task ?? {}} />
        </TabsContent>
        <TabsContent value="model" className="mt-3">
          <BucketTable title="Per-model cost" buckets={s?.per_model ?? {}} />
        </TabsContent>
        <TabsContent value="recent" className="mt-3">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Last {r.length} calls
              </CardTitle>
              <div className="flex gap-1">
                {[20, 50, 100, 200].map((n) => (
                  <Button
                    key={n}
                    size="sm"
                    variant={recentLimit === n ? 'default' : 'outline'}
                    className="h-7 px-2 text-xs"
                    onClick={() => setRecentLimit(n)}
                  >
                    {n}
                  </Button>
                ))}
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <ScrollArea className="h-[420px]">
                <table className="w-full text-sm">
                  <thead className="bg-muted/40 text-xs uppercase sticky top-0">
                    <tr>
                      <th className="text-left px-4 py-2 font-medium">Time</th>
                      <th className="text-left px-4 py-2 font-medium">Task</th>
                      <th className="text-left px-4 py-2 font-medium">Model</th>
                      <th className="text-right px-4 py-2 font-medium">ms</th>
                      <th className="text-right px-4 py-2 font-medium">In</th>
                      <th className="text-right px-4 py-2 font-medium">Out</th>
                      <th className="text-right px-4 py-2 font-medium">USD</th>
                      <th className="text-left px-4 py-2 font-medium">Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="text-center text-muted-foreground py-8">
                          No calls yet
                        </td>
                      </tr>
                    ) : (
                      r.map((e, i) => (
                        <tr key={`${e.id}-${i}`} className="border-t">
                          <td className="px-4 py-1.5 text-xs tabular-nums">{fmtTs(e.timestamp)}</td>
                          <td className="px-4 py-1.5 font-mono text-xs">{e.task}</td>
                          <td className="px-4 py-1.5 font-mono text-xs">{e.model}</td>
                          <td className="text-right tabular-nums px-4 py-1.5">{e.latency_ms}</td>
                          <td className="text-right tabular-nums px-4 py-1.5">{fmtInt(e.in_tokens)}</td>
                          <td className="text-right tabular-nums px-4 py-1.5">{fmtInt(e.out_tokens)}</td>
                          <td className="text-right tabular-nums px-4 py-1.5">{fmtUsd(e.usd)}</td>
                          <td className="px-4 py-1.5">
                            <Badge variant="outline" className="text-[10px]">
                              {e.usage_source}
                            </Badge>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="charts" className="mt-3">
          <CostCharts perModel={s?.per_model ?? {}} recent={r ?? []} />
        </TabsContent>
      </Tabs>
        </div>
      </div>
    </AppShell>
  )
}
