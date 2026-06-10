'use client'

import * as HoverCard from '@radix-ui/react-hover-card'
import { FileText, Image as ImageIcon, Layers } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Badge } from '@/components/ui/badge'
import { useTranslation } from '@/lib/hooks/use-translation'
import type { Citation } from '@/lib/types/api'

interface CitationHoverCardProps {
  index: number
  citation: Citation
  children: React.ReactNode
}

export function RaptorBadge({ nodeLevel }: { nodeLevel?: number | null }) {
  const { t } = useTranslation()
  if (!nodeLevel || nodeLevel < 1) return null
  return (
    <Badge variant="outline" className="text-[10px] px-1 py-0 gap-0.5 text-violet-600 dark:text-violet-300 border-violet-300/50">
      <Layers className="h-2.5 w-2.5" />{t('ragSignals.summary')} · L{nodeLevel}
    </Badge>
  )
}

export function CitationContextLine({ text }: { text?: string | null }) {
  const t = (text || '').trim()
  if (!t) return null
  return <div className="text-[10px] text-muted-foreground italic mb-1 line-clamp-2">{t}</div>
}

function renderExcerptBody(citation: Citation) {
  if (citation.segment_type === 'image' && citation.image_url) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={citation.image_url}
        alt={citation.section_path || citation.document_title}
        className="max-h-40 max-w-full rounded border border-border"
        loading="lazy"
      />
    )
  }
  const excerpt = (citation.excerpt || '').trim()
  if (!excerpt) {
    return <span className="text-muted-foreground italic text-xs">No excerpt available</span>
  }
  return (
    <div className="prose prose-xs prose-neutral dark:prose-invert max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { strict: 'ignore', throwOnError: false }]]}
      >
        {excerpt}
      </ReactMarkdown>
    </div>
  )
}

export function CitationHoverCard({ index, citation, children }: CitationHoverCardProps) {
  const isImage = citation.segment_type === 'image'
  return (
    <HoverCard.Root openDelay={120} closeDelay={80}>
      <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="top"
          sideOffset={6}
          className="z-50 w-[380px] max-w-[90vw] rounded-md border bg-popover p-3 shadow-md text-popover-foreground"
        >
          <div className="flex items-start gap-2 mb-2">
            {isImage ? (
              <ImageIcon className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            ) : (
              <FileText className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            )}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-[10px] font-mono text-muted-foreground">[{index}]</span>
                <span className="text-xs font-medium truncate">{citation.document_title}</span>
                {citation.page_label && (
                  <Badge variant="outline" className="text-[10px] px-1 py-0">
                    {citation.page_label}
                  </Badge>
                )}
                <RaptorBadge nodeLevel={citation.node_level} />
              </div>
              {citation.section_path && (
                <div className="text-[10px] text-muted-foreground truncate">
                  {citation.section_path}
                </div>
              )}
            </div>
          </div>
          <div className="border-t pt-2">
            <CitationContextLine text={citation.context_text} />
            {renderExcerptBody(citation)}
          </div>
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  )
}

export default CitationHoverCard
