'use client'

import type { Citation } from '@/lib/types/api'

interface InlineImageCitationProps {
  citation: Citation
}

export function InlineImageCitation({ citation }: InlineImageCitationProps) {
  if (!citation.image_url) return null
  return (
    <aside className="float-right ml-3 mb-2 w-44 rounded border border-border bg-muted/40 p-1 text-[10px]">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={citation.image_url}
        alt={citation.section_path || citation.document_title}
        loading="lazy"
        className="w-full h-auto rounded"
      />
      <div className="mt-1 px-0.5 text-muted-foreground leading-tight line-clamp-2">
        {citation.section_path || citation.document_title}
        {citation.page_label ? ` · ${citation.page_label}` : ''}
      </div>
    </aside>
  )
}

export default InlineImageCitation
