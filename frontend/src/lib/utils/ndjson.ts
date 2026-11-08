/**
 * Incremental reader for newline-delimited JSON streams (also accepts SSE
 * `data: ` lines).
 *
 * Network chunks do not respect character or line boundaries. Decoding each
 * chunk on its own turns a multi-byte UTF-8 character cut in half (common in
 * Vietnamese) into U+FFFD, and splitting each chunk on '\n' silently drops a
 * JSON line cut in half. This keeps decoder state and the unfinished line
 * between chunks.
 */
export function createNdjsonReader() {
  const decoder = new TextDecoder()
  let buffer = ''

  return function read(chunk: Uint8Array | undefined, end = false): string[] {
    buffer += chunk ? decoder.decode(chunk, { stream: !end }) : decoder.decode()
    const parts = buffer.split('\n')
    buffer = end ? '' : (parts.pop() ?? '')
    return parts
      .map(line => (line.startsWith('data: ') ? line.slice(6) : line).trim())
      .filter(line => line.length > 0)
  }
}
