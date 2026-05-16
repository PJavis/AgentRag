import apiClient from './client'

export interface CostBucket {
  calls: number
  in_tokens: number
  out_tokens: number
  total_latency_ms: number
  avg_latency_ms: number
  p50_latency_ms?: number
  p95_latency_ms?: number
  usd: number
}

export interface CostSummary {
  total_calls: number
  total_in_tokens: number
  total_out_tokens: number
  total_usd: number
  per_task: Record<string, CostBucket>
  per_model: Record<string, CostBucket>
  note: string
}

export interface CostEntry {
  id: number
  timestamp: number
  task: string
  model: string
  latency_ms: number
  in_tokens: number
  out_tokens: number
  usd: number
  usage_source: 'provider' | 'estimate'
}

export const metricsApi = {
  summary: async (): Promise<CostSummary> => {
    const r = await apiClient.get<CostSummary>('/metrics/cost')
    return r.data
  },
  recent: async (limit = 50, since?: number): Promise<CostEntry[]> => {
    const params: Record<string, string | number> = { limit }
    if (since !== undefined) params.since = since
    const r = await apiClient.get<{ entries: CostEntry[] }>('/metrics/cost/recent', { params })
    return r.data.entries
  },
  reset: async (): Promise<void> => {
    await apiClient.post('/metrics/cost/reset')
  },
}

export default metricsApi
