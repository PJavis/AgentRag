'use client'

import { FileText, Loader2, CheckCircle2, XCircle, Clock, RefreshCw } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { IngestProgressItem } from '@/lib/api/activity'

interface IngestProgressProps {
  items: IngestProgressItem[]
  activeCount: number
  loading?: boolean
  includeDone: boolean
  onToggleIncludeDone: () => void
  emptyText?: string
}

function StatusBadge({ status }: { status: IngestProgressItem['graph_status'] }) {
  const map = {
    processing: { Icon: Loader2, cls: 'bg-blue-500/15 text-blue-700 dark:text-blue-300 border-blue-500/30', label: 'Đang xử lý', spin: true },
    queued: { Icon: Clock, cls: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30', label: 'Trong hàng đợi', spin: false },
    pending: { Icon: Clock, cls: 'bg-slate-500/15 text-slate-700 dark:text-slate-300 border-slate-500/30', label: 'Chờ xử lý', spin: false },
    failed: { Icon: XCircle, cls: 'bg-red-500/15 text-red-700 dark:text-red-300 border-red-500/30', label: 'Lỗi', spin: false },
    done: { Icon: CheckCircle2, cls: 'bg-green-500/15 text-green-700 dark:text-green-300 border-green-500/30', label: 'Hoàn tất', spin: false },
  } as const
  const cfg = map[status as keyof typeof map] ?? map.pending
  const { Icon } = cfg
  return (
    <Badge variant="outline" className={`gap-1 font-normal ${cfg.cls}`}>
      <Icon className={`h-3 w-3 ${cfg.spin ? 'animate-spin' : ''}`} />
      {cfg.label}
    </Badge>
  )
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  const diffSec = (Date.now() - d.getTime()) / 1000
  if (diffSec < 60) return `${Math.floor(diffSec)}s trước`
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m trước`
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h trước`
  return `${Math.floor(diffSec / 86400)}d trước`
}

function ProgressBar({ pct, status }: { pct: number; status: IngestProgressItem['graph_status'] }) {
  const colour =
    status === 'failed' ? 'bg-red-500' :
    status === 'done' ? 'bg-green-500' :
    status === 'processing' ? 'bg-blue-500' :
    'bg-slate-400'
  return (
    <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
      <div
        className={`h-full transition-all duration-500 ${colour}`}
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  )
}

export function IngestProgress({
  items,
  activeCount,
  loading,
  includeDone,
  onToggleIncludeDone,
  emptyText = 'Không có file nào đang xử lý.',
}: IngestProgressProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm">
          {activeCount > 0 ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />
              <span className="text-muted-foreground">
                <span className="font-medium text-foreground">{activeCount}</span> file đang xử lý
              </span>
            </>
          ) : (
            <span className="text-muted-foreground text-xs">{emptyText}</span>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs gap-1"
          onClick={onToggleIncludeDone}
        >
          <RefreshCw className="h-3 w-3" />
          {includeDone ? 'Ẩn đã xong' : 'Hiện đã xong'}
        </Button>
      </div>

      {loading && items.length === 0 && (
        <div className="text-xs text-muted-foreground text-center py-4">Đang tải…</div>
      )}

      <div className="space-y-2">
        {items.map((item) => (
          <div
            key={item.document_id}
            className="border rounded-lg p-3 space-y-2 hover:bg-accent/30 transition-colors"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-2 min-w-0 flex-1">
                <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0 mt-0.5" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium truncate" title={item.title}>
                    {item.title}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                    <span className="uppercase">{item.source_type || 'unknown'}</span>
                    <span>·</span>
                    <span>{formatRelativeTime(item.updated_at || item.created_at)}</span>
                  </div>
                </div>
              </div>
              <StatusBadge status={item.graph_status} />
            </div>

            {(item.chunks_total > 0 || item.graph_status === 'processing') && (
              <>
                <ProgressBar pct={item.progress_pct} status={item.graph_status} />
                <div className="flex items-center justify-between text-xs text-muted-foreground tabular-nums">
                  <span>
                    {item.chunks_done.toLocaleString()} / {item.chunks_total.toLocaleString()} chunks
                    {item.chunks_failed > 0 && (
                      <span className="ml-2 text-red-600 dark:text-red-400">
                        · {item.chunks_failed} lỗi
                      </span>
                    )}
                  </span>
                  <span className="font-medium">{item.progress_pct.toFixed(0)}%</span>
                </div>
              </>
            )}

            {item.graph_status === 'failed' && item.graph_last_error && (
              <div className="text-xs text-red-600 dark:text-red-400 bg-red-500/10 rounded px-2 py-1 truncate" title={item.graph_last_error}>
                {item.graph_last_error}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
