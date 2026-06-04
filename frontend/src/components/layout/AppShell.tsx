'use client'

import { AppSidebar } from './AppSidebar'
import { SetupBanner } from './SetupBanner'
import { useIngestProgress } from '@/lib/hooks/useIngestProgress'

interface AppShellProps {
  children: React.ReactNode
}

export function AppShell({ children }: AppShellProps) {
  // One global ingest-progress SSE subscription → source lists refresh live.
  useIngestProgress()
  return (
    <div className="flex h-screen overflow-hidden">
      <AppSidebar />
      <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <SetupBanner />
        {children}
      </main>
    </div>
  )
}
