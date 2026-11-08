import { sourcesApi } from '@/lib/api/sources'

const BLOB_URL_TTL_MS = 15 * 60_000

/** First page of a citation's range: page_start, else a numeric `page`. */
export function citationPage(citation: { page_start?: number | null; page?: unknown }): number | null {
  if (typeof citation.page_start === 'number' && citation.page_start > 0) return citation.page_start
  const n = typeof citation.page === 'number' ? citation.page : parseInt(String(citation.page ?? ''), 10)
  return Number.isFinite(n) && n > 0 ? n : null
}

/** Blob URL with a `#page=N` fragment — honoured by the browser PDF viewers. */
export function pageUrl(blobUrl: string, page: number | null): string {
  return page ? `${blobUrl}#page=${page}` : blobUrl
}

/**
 * Open a source's original file in a new tab, at `page` when it is a PDF.
 * Returns false when there is no viewable original (missing file or not a
 * PDF) so the caller can fall back to the source detail view.
 *
 * The download needs the bearer token, so the file is fetched through the API
 * client and opened as a blob URL — a plain link would get 401.
 */
export async function openOriginalAtPage(
  sourceId: string,
  page: number | null,
  sourceType?: string | null,
): Promise<boolean> {
  // Known non-PDF: no page to open — skip downloading the whole file (and
  // flashing a blank tab) just to learn that from the blob type.
  if (sourceType && sourceType.toLowerCase().replace(/^\./, '') !== 'pdf') return false
  // Open the tab synchronously (inside the click) so popup blockers allow it,
  // then point it at the blob once the download finishes.
  const tab = window.open('', '_blank')
  try {
    const response = await sourcesApi.downloadFile(sourceId)
    const blob = response.data
    if (!blob || blob.type !== 'application/pdf') {
      tab?.close()
      return false
    }
    const blobUrl = window.URL.createObjectURL(blob)
    if (tab) tab.location.href = pageUrl(blobUrl, page)
    else window.open(pageUrl(blobUrl, page), '_blank')
    // Long enough for a large PDF to load and for in-tab reloads; a blob URL
    // cannot outlive this page anyway.
    setTimeout(() => window.URL.revokeObjectURL(blobUrl), BLOB_URL_TTL_MS)
    return true
  } catch {
    tab?.close()
    return false
  }
}
