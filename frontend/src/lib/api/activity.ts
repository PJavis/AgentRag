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
}

export default activityApi
