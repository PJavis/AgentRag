'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { metricsApi, CostSummary, CostEntry } from '@/lib/api/metrics'

export const COST_QUERY_KEYS = {
  summary: ['metrics', 'cost', 'summary'] as const,
  recent: (limit: number) => ['metrics', 'cost', 'recent', limit] as const,
}

export function useCostSummary(refetchIntervalMs = 5000) {
  return useQuery<CostSummary>({
    queryKey: COST_QUERY_KEYS.summary,
    queryFn: metricsApi.summary,
    refetchInterval: refetchIntervalMs,
    refetchOnWindowFocus: true,
  })
}

export function useRecentCostCalls(limit = 50, refetchIntervalMs = 5000) {
  return useQuery<CostEntry[]>({
    queryKey: COST_QUERY_KEYS.recent(limit),
    queryFn: () => metricsApi.recent(limit),
    refetchInterval: refetchIntervalMs,
  })
}

export function useResetCostLedger() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: metricsApi.reset,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['metrics', 'cost'] })
      toast.success('Cost ledger cleared')
    },
    onError: () => toast.error('Failed to reset ledger'),
  })
}
