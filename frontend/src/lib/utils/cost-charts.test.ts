import { describe, it, expect } from 'vitest'
import { perModelSpend, costOverTime } from './cost-charts'
import type { CostBucket, CostEntry } from '@/lib/api/metrics'

const bucket = (usd: number, calls = 1): CostBucket => ({
  calls, in_tokens: 0, out_tokens: 0, total_latency_ms: 0, avg_latency_ms: 0, usd,
})

describe('perModelSpend', () => {
  it('returns sorted [{name, usd, calls}] descending by usd', () => {
    const out = perModelSpend({ 'gpt-4': bucket(2, 3), 'mini': bucket(5, 9) })
    expect(out[0]).toEqual({ name: 'mini', usd: 5, calls: 9 })
    expect(out[1].name).toBe('gpt-4')
  })
  it('empty input → empty array', () => {
    expect(perModelSpend({})).toEqual([])
  })
})

describe('costOverTime', () => {
  it('buckets entries by minute and accumulates usd', () => {
    const entries: CostEntry[] = [
      { id: 1, timestamp: 60, task: 'chat', model: 'm', latency_ms: 0, in_tokens: 0, out_tokens: 0, usd: 0.1, usage_source: 'estimate' },
      { id: 2, timestamp: 90, task: 'chat', model: 'm', latency_ms: 0, in_tokens: 0, out_tokens: 0, usd: 0.2, usage_source: 'estimate' },
      { id: 3, timestamp: 130, task: 'chat', model: 'm', latency_ms: 0, in_tokens: 0, out_tokens: 0, usd: 0.4, usage_source: 'estimate' },
    ]
    const out = costOverTime(entries)
    expect(out).toHaveLength(2)
    expect(out[0].usd).toBeCloseTo(0.3)
    expect(out[1].usd).toBeCloseTo(0.4)
  })
  it('empty → empty', () => {
    expect(costOverTime([])).toEqual([])
  })
})
