import apiClient from './client'
import {
  ActivityEvent,
  ActivitySummary,
  AdminUserEntry,
} from '@/lib/types/api'

export interface EventsParams {
  type?: string
  limit?: number
  before_id?: string
}

export interface AdminEventsParams extends EventsParams {
  user_id?: string
}

export interface EventsPage {
  entries: ActivityEvent[]
  next_before_id: string | null
}

export interface IngestProgressItem {
  document_id: string
  title: string
  source_type: string | null
  graph_status: 'pending' | 'queued' | 'processing' | 'failed' | 'done' | null
  progress_pct: number
  chunks_total: number
  chunks_done: number
  chunks_failed: number
  graph_last_error: string | null
  created_at: string | null
  updated_at: string | null
}

export interface IngestProgressPage {
  items: IngestProgressItem[]
  active_count: number
}

export const activityApi = {
  summary: async (): Promise<ActivitySummary> =>
    (await apiClient.get<ActivitySummary>('/activity/summary')).data,
  events: async (params: EventsParams): Promise<EventsPage> =>
    (await apiClient.get<EventsPage>('/activity/events', { params })).data,
  event: async (id: string): Promise<ActivityEvent> =>
    (await apiClient.get<ActivityEvent>(`/activity/events/${id}`)).data,
  adminSummary: async (userId?: string): Promise<ActivitySummary> =>
    (await apiClient.get<ActivitySummary>('/admin/activity/summary', {
      params: userId ? { user_id: userId } : {},
    })).data,
  adminEvents: async (params: AdminEventsParams): Promise<EventsPage> =>
    (await apiClient.get<EventsPage>('/admin/activity/events', { params })).data,
  adminUsers: async (): Promise<AdminUserEntry[]> =>
    (await apiClient.get<AdminUserEntry[]>('/admin/activity/users')).data,
  ingestProgress: async (includeDone = false, limit = 30): Promise<IngestProgressPage> =>
    (await apiClient.get<IngestProgressPage>('/activity/ingest-progress', {
      params: { include_done: includeDone, limit },
    })).data,
  adminIngestProgress: async (userId?: string, includeDone = false, limit = 50): Promise<IngestProgressPage> =>
    (await apiClient.get<IngestProgressPage>('/admin/activity/ingest-progress', {
      params: {
        ...(userId ? { user_id: userId } : {}),
        include_done: includeDone,
        limit,
      },
    })).data,
}

export default activityApi
