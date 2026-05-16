'use client'

import { useMemo, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ChevronRight, Activity, Clock, ListTree } from 'lucide-react'
import type {
  NotebookChatMessage,
  ChatTimings,
  ToolTraceEntry,
} from '@/lib/types/api'

interface TraceDialogProps {
  open: boolean
  onOpenChange: (v: boolean) => void
  message: NotebookChatMessage | null
}

type StageKey =
  | 'plan'
  | 'decide'
  | 'tool'
  | 'assemble'
  | 'answer'
  | 'critique'

const STAGE_ORDER: StageKey[] = ['plan', 'decide', 'tool', 'assemble', 'answer', 'critique']
const STAGE_LABEL: Record<StageKey, string> = {
  plan: 'Plan',
  decide: 'Decide',
  tool: 'Tool',
  assemble: 'Assemble',
  answer: 'Answer',
  critique: 'Critique',
}

function stageEntries(timings?: ChatTimings) {
  return STAGE_ORDER.map((k) => ({
    key: k,
    label: STAGE_LABEL[k],
    ms: timings?.[k],
  })).filter((e) => typeof e.ms === 'number' && (e.ms as number) > 0)
}

function StageNode({
  label,
  ms,
  active,
  onClick,
}: {
  label: string
  ms?: number
  active?: boolean
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-col items-start rounded-md border px-3 py-2 text-left transition ${
        active
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/40'
      }`}
    >
      <span className="text-xs font-medium">{label}</span>
      {typeof ms === 'number' && (
        <span className="text-[10px] font-mono text-muted-foreground tabular-nums">
          {ms.toFixed(0)} ms
        </span>
      )}
    </button>
  )
}

function ToolTraceList({ entries }: { entries: ToolTraceEntry[] }) {
  const [expanded, setExpanded] = useState<number | null>(null)
  if (!entries?.length) {
    return <div className="text-xs text-muted-foreground">No tool calls.</div>
  }
  return (
    <div className="space-y-2">
      {entries.map((e, i) => {
        const isOpen = expanded === i
        const name = e.tool_name || 'unknown'
        const subq = (e.sub_query as string | undefined) ?? null
        const inMs = e.tool_latency_ms ?? 0
        const decMs = e.decision_latency_ms ?? 0
        return (
          <div key={i} className="rounded border">
            <button
              onClick={() => setExpanded(isOpen ? null : i)}
              className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-muted/40"
            >
              <div className="flex items-center gap-2 min-w-0">
                <ChevronRight
                  className={`h-3 w-3 shrink-0 transition ${isOpen ? 'rotate-90' : ''}`}
                />
                <span className="text-xs font-mono shrink-0">{name}</span>
                {subq && (
                  <span className="text-[10px] text-muted-foreground truncate">
                    “{subq}”
                  </span>
                )}
              </div>
              <div className="flex gap-1 shrink-0">
                {decMs > 0 && (
                  <Badge variant="outline" className="text-[10px] font-mono">
                    dec {decMs.toFixed(0)}ms
                  </Badge>
                )}
                <Badge variant="secondary" className="text-[10px] font-mono">
                  {inMs.toFixed(0)}ms
                </Badge>
              </div>
            </button>
            {isOpen && (
              <div className="border-t bg-muted/20 px-3 py-2 space-y-2">
                <div>
                  <div className="text-[10px] uppercase text-muted-foreground mb-1">
                    Input
                  </div>
                  <pre className="text-[11px] font-mono whitespace-pre-wrap break-all bg-background rounded p-2 max-h-40 overflow-auto">
                    {JSON.stringify(e.tool_input ?? {}, null, 2)}
                  </pre>
                </div>
                <div>
                  <div className="text-[10px] uppercase text-muted-foreground mb-1">
                    Output (truncated)
                  </div>
                  <pre className="text-[11px] font-mono whitespace-pre-wrap break-all bg-background rounded p-2 max-h-60 overflow-auto">
                    {JSON.stringify(e.tool_output ?? {}, null, 2).slice(0, 4000)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function TraceDialog({ open, onOpenChange, message }: TraceDialogProps) {
  const [selectedStage, setSelectedStage] = useState<StageKey | null>(null)
  const stages = useMemo(() => stageEntries(message?.timings_ms), [message])
  const total = message?.timings_ms?.total ?? 0
  const path = message?.reasoning_path ?? 'unknown'
  const tools = message?.tool_trace ?? []
  const subqs = (message?.plan_subqueries ?? []) as Array<string | Record<string, unknown>>
  const sql = message?.sql_query ?? null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ListTree className="h-4 w-4" />
            Reasoning trace
            <Badge variant="outline" className="font-mono text-[10px]">
              {path}
            </Badge>
            <Badge variant="secondary" className="font-mono text-[10px] gap-1">
              <Clock className="h-3 w-3" />
              {total.toFixed(0)} ms total
            </Badge>
          </DialogTitle>
        </DialogHeader>

        <ScrollArea className="flex-1 -mx-6 px-6">
          <div className="space-y-5 py-2">
            {/* Stage graph */}
            <div>
              <div className="text-xs font-medium mb-2 flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" />
                Pipeline
              </div>
              {stages.length === 0 ? (
                <div className="text-xs text-muted-foreground">
                  No timing data available.
                </div>
              ) : (
                <div className="flex items-center gap-1 flex-wrap">
                  {stages.map((s, i) => (
                    <div key={s.key} className="flex items-center gap-1">
                      <StageNode
                        label={s.label}
                        ms={s.ms}
                        active={selectedStage === s.key}
                        onClick={() =>
                          setSelectedStage(selectedStage === s.key ? null : s.key)
                        }
                      />
                      {i < stages.length - 1 && (
                        <ChevronRight className="h-3 w-3 text-muted-foreground" />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Sub-queries */}
            {subqs.length > 0 && (
              <div>
                <div className="text-xs font-medium mb-2">
                  Plan ({subqs.length} sub-queries)
                </div>
                <ul className="space-y-1">
                  {subqs.map((q, i) => (
                    <li
                      key={i}
                      className="text-xs font-mono px-3 py-1.5 rounded bg-muted/40 border"
                    >
                      <span className="text-muted-foreground mr-2">{i + 1}.</span>
                      {typeof q === 'string'
                        ? q
                        : ((q as Record<string, unknown>).query as string) ||
                          JSON.stringify(q)}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* SQL */}
            {sql && (
              <div>
                <div className="text-xs font-medium mb-2">Generated SQL</div>
                <pre className="text-[11px] font-mono whitespace-pre-wrap break-all bg-muted/40 rounded p-2 border">
                  {sql}
                </pre>
              </div>
            )}

            {/* Tool calls */}
            <div>
              <div className="text-xs font-medium mb-2">
                Tool calls ({tools.length})
              </div>
              <ToolTraceList entries={tools} />
            </div>

            {/* Citations */}
            {message?.citations && message.citations.length > 0 && (
              <div>
                <div className="text-xs font-medium mb-2">
                  Citations ({message.citations.length})
                </div>
                <ul className="space-y-1">
                  {message.citations.map((c, i) => {
                    const obj = c as Record<string, unknown>
                    return (
                      <li
                        key={i}
                        className="text-xs px-3 py-1.5 rounded border bg-muted/20"
                      >
                        <span className="font-mono text-[10px] text-muted-foreground mr-2">
                          #{i + 1}
                        </span>
                        <span className="font-medium">
                          {(obj.document_title as string) || 'unknown'}
                        </span>
                        {obj.section_path != null && (
                          <span className="text-muted-foreground ml-2">
                            / {String(obj.section_path)}
                          </span>
                        )}
                      </li>
                    )
                  })}
                </ul>
              </div>
            )}
          </div>
        </ScrollArea>

        <div className="pt-2 flex justify-end">
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default TraceDialog
