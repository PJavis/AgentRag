'use client'
import { useQuery } from '@tanstack/react-query'
import { activityApi } from '@/lib/api/activity'

export const ACTIVITY_QUERY_KEYS = {
  summary: ['activity', 'summary'] as const,
  events: (type?: string, limit?: number) =>
    ['activity', 'events', type ?? 'all', limit ?? 50] as const,
  adminSummary: (userId?: string) =>
    ['admin', 'activity', 'summary', userId ?? 'all'] as const,
  adminEvents: (userId?: string, type?: string) =>
    ['admin', 'activity', 'events', userId ?? 'all', type ?? 'all'] as const,
  adminUsers: ['admin', 'activity', 'users'] as const,
  ingestProgress: (includeDone?: boolean) =>
    ['activity', 'ingest-progress', includeDone ? 'with-done' : 'active'] as const,
  adminIngestProgress: (userId?: string, includeDone?: boolean) =>
    ['admin', 'activity', 'ingest-progress', userId ?? 'all', includeDone ? 'with-done' : 'active'] as const,
}

export function useActivitySummary(refetchMs = 30_000) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.summary,
    queryFn: activityApi.summary,
    refetchInterval: refetchMs,
  })
}

export function useActivityEvents(type?: string, limit = 50, refetchMs = 30_000) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.events(type, limit),
    queryFn: () => activityApi.events({ type, limit }),
    refetchInterval: refetchMs,
  })
}

export function useAdminActivitySummary(userId?: string) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.adminSummary(userId),
    queryFn: () => activityApi.adminSummary(userId),
  })
}

export function useAdminActivityEvents(userId?: string, type?: string, limit = 50) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.adminEvents(userId, type),
    queryFn: () => activityApi.adminEvents({ user_id: userId, type, limit }),
  })
}

export function useAdminUsers() {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.adminUsers,
    queryFn: activityApi.adminUsers,
  })
}

export function useIngestProgress(includeDone = false, refetchMs = 4_000) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.ingestProgress(includeDone),
    queryFn: () => activityApi.ingestProgress(includeDone),
    refetchInterval: refetchMs,
  })
}

export function useAdminIngestProgress(userId?: string, includeDone = false, refetchMs = 4_000) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.adminIngestProgress(userId, includeDone),
    queryFn: () => activityApi.adminIngestProgress(userId, includeDone),
    refetchInterval: refetchMs,
  })
}
