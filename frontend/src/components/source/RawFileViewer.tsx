'use client'

import { useEffect, useMemo, useState } from 'react'
import { Loader2, AlertCircle } from 'lucide-react'
import { sourcesApi } from '@/lib/api/sources'

interface RawFileViewerProps {
  sourceId: string
  ext: string
  className?: string
}

const INLINE_PDF = new Set(['pdf'])
const INLINE_IMAGE = new Set(['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'])
const INLINE_TEXT = new Set(['md', 'txt', 'csv', 'json', 'html'])

export function isInlineRawSupported(ext?: string | null): boolean {
  if (!ext) return false
  const e = ext.toLowerCase().replace(/^\./, '')
  return INLINE_PDF.has(e) || INLINE_IMAGE.has(e) || INLINE_TEXT.has(e)
}

export function RawFileViewer({ sourceId, ext, className }: RawFileViewerProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [textContent, setTextContent] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const safeExt = useMemo(() => ext.toLowerCase().replace(/^\./, ''), [ext])
  const kind: 'pdf' | 'image' | 'text' | 'unknown' = useMemo(() => {
    if (INLINE_PDF.has(safeExt)) return 'pdf'
    if (INLINE_IMAGE.has(safeExt)) return 'image'
    if (INLINE_TEXT.has(safeExt)) return 'text'
    return 'unknown'
  }, [safeExt])

  useEffect(() => {
    let cancelled = false
    let createdUrl: string | null = null
    setLoading(true)
    setError(null)
    setBlobUrl(null)
    setTextContent(null)

    sourcesApi
      .downloadFile(sourceId)
      .then(async (response) => {
        if (cancelled) return
        if (kind === 'text') {
          const text = await response.data.text()
          if (!cancelled) setTextContent(text)
        } else {
          createdUrl = window.URL.createObjectURL(response.data)
          if (!cancelled) setBlobUrl(createdUrl)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          const status = (err as { response?: { status?: number } })?.response?.status
          setError(status === 404 ? 'Original file not available' : 'Failed to load file')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
      if (createdUrl) window.URL.revokeObjectURL(createdUrl)
    }
  }, [sourceId, kind])

  if (loading) {
    return (
      <div className={`flex items-center justify-center py-20 text-muted-foreground ${className ?? ''}`}>
        <Loader2 className="h-5 w-5 animate-spin mr-2" />
        Loading original file…
      </div>
    )
  }

  if (error) {
    return (
      <div className={`flex items-center gap-2 text-sm text-destructive ${className ?? ''}`}>
        <AlertCircle className="h-4 w-4" />
        {error}
      </div>
    )
  }

  if (kind === 'pdf' && blobUrl) {
    return (
      <iframe
        src={blobUrl}
        title="Original PDF"
        className={`w-full h-[80vh] rounded border border-border bg-background ${className ?? ''}`}
      />
    )
  }

  if (kind === 'image' && blobUrl) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={blobUrl}
        alt="Original"
        className={`max-w-full h-auto rounded border border-border ${className ?? ''}`}
      />
    )
  }

  if (kind === 'text' && textContent !== null) {
    return (
      <pre className={`whitespace-pre-wrap break-words text-sm font-mono bg-muted/30 rounded border p-3 max-h-[80vh] overflow-auto ${className ?? ''}`}>
        {textContent}
      </pre>
    )
  }

  return (
    <div className={`text-sm text-muted-foreground ${className ?? ''}`}>
      Inline preview not supported for .{safeExt}. Use the &quot;Open original&quot; button to download.
    </div>
  )
}

export default RawFileViewer
