'use client'

import { Zap, Brain, Database, RotateCcw, Hospital, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { useTranslation } from '@/lib/hooks/use-translation'
import type { NotebookChatMessage, SourceChatMessage } from '@/lib/types/api'

type ChatMsg = NotebookChatMessage | SourceChatMessage

export function deriveCritiqueState(message: ChatMsg): 'verified' | 'corrected' | null {
  const ranCritique = typeof message.timings_ms?.critique === 'number'
  if (!ranCritique) return null
  const corrected = (message.tool_trace ?? []).some(
    (e) => (e as Record<string, unknown>).corrective === true,
  )
  return corrected ? 'corrected' : 'verified'
}

const PATH_META: Record<string, { labelKey: string; icon: typeof Zap }> = {
  fast: { labelKey: 'ragSignals.fastPath', icon: Zap },
  semantic: { labelKey: 'ragSignals.semantic', icon: Brain },
  structured: { labelKey: 'ragSignals.structured', icon: Database },
}

export function MessageSignals({ message }: { message: ChatMsg }) {
  const { t } = useTranslation()
  const path = message.reasoning_path || ''
  const pathMeta = PATH_META[path]
  const critique = deriveCritiqueState(message)
  const chips: React.ReactNode[] = []

  if (pathMeta) {
    const Icon = pathMeta.icon
    chips.push(
      <Badge key="path" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in"
        title={t('ragSignals.fastPathTitle')}>
        <Icon className="h-3 w-3" />{t(pathMeta.labelKey)}
      </Badge>,
    )
  } else if (path) {
    chips.push(
      <Badge key="path" variant="outline" className="text-[10px] h-5 font-mono animate-in fade-in">
        {path}
      </Badge>,
    )
  }

  if (message.semantic_cache_hit) {
    chips.push(
      <Badge key="cache" variant="secondary"
        className="text-[10px] h-5 gap-1 animate-in fade-in text-amber-700 dark:text-amber-300"
        title={t('ragSignals.cacheTitle')}>
        <Sparkles className="h-3 w-3" />{t('ragSignals.instantCached')}
      </Badge>,
    )
  }

  if (critique === 'verified') {
    chips.push(
      <Badge key="crit" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in text-emerald-700 dark:text-emerald-300 border-emerald-300/50"
        title={t('ragSignals.verifiedTitle')}>
        <Brain className="h-3 w-3" />{t('ragSignals.verified')}
      </Badge>,
    )
  } else if (critique === 'corrected') {
    chips.push(
      <Badge key="crit" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in text-blue-700 dark:text-blue-300 border-blue-300/50"
        title={t('ragSignals.correctedTitle')}>
        <RotateCcw className="h-3 w-3" />{t('ragSignals.selfCorrected')}
      </Badge>,
    )
  }

  if (message.domain_route) {
    chips.push(
      <Badge key="domain" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in"
        title={t('ragSignals.domainTitle')}>
        <Hospital className="h-3 w-3" />{message.domain_route}
      </Badge>,
    )
  }

  if (chips.length === 0) return null
  return <div className="flex items-center gap-1 flex-wrap">{chips}</div>
}

export default MessageSignals
