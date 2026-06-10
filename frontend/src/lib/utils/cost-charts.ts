import type { CostBucket, CostEntry } from '@/lib/api/metrics'

export interface ModelSpend { name: string; usd: number; calls: number }
export interface TimePoint { t: number; label: string; usd: number; tokens: number }

export function perModelSpend(perModel: Record<string, CostBucket>): ModelSpend[] {
  return Object.entries(perModel)
    .map(([name, b]) => ({ name, usd: b.usd, calls: b.calls }))
    .sort((a, b) => b.usd - a.usd)
}

export function costOverTime(entries: CostEntry[], bucketSec = 60): TimePoint[] {
  if (!entries.length) return []
  const buckets = new Map<number, TimePoint>()
  for (const e of entries) {
    const key = Math.floor(e.timestamp / bucketSec) * bucketSec
    const prev = buckets.get(key)
    const label = new Date(key * 1000).toLocaleTimeString(undefined, { hour12: false })
    if (prev) {
      prev.usd += e.usd
      prev.tokens += e.in_tokens + e.out_tokens
    } else {
      buckets.set(key, { t: key, label, usd: e.usd, tokens: e.in_tokens + e.out_tokens })
    }
  }
  return Array.from(buckets.values()).sort((a, b) => a.t - b.t)
}
