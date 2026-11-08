import { describe, expect, it } from 'vitest'
import { createNdjsonReader } from './ndjson'

const enc = new TextEncoder()

describe('createNdjsonReader', () => {
  it('reassembles a Vietnamese character split across chunks', () => {
    const bytes = enc.encode('{"content":"hiếm muộn"}\n')
    // Cut inside the 3-byte "ế".
    const cut = bytes.indexOf(0xe1) + 1
    const read = createNdjsonReader()
    const first = read(bytes.slice(0, cut))
    const second = read(bytes.slice(cut))
    expect(first).toEqual([])
    expect(second).toEqual(['{"content":"hiếm muộn"}'])
    expect(JSON.parse(second[0]).content).not.toContain('�')
  })

  it('keeps a JSON line split across chunks instead of dropping it', () => {
    const read = createNdjsonReader()
    expect(read(enc.encode('{"type":"ai","con'))).toEqual([])
    expect(read(enc.encode('tent":"a"}\n{"type":"done"}\n'))).toEqual([
      '{"type":"ai","content":"a"}',
      '{"type":"done"}',
    ])
  })

  it('strips an SSE data: prefix and skips blank lines', () => {
    const read = createNdjsonReader()
    expect(read(enc.encode('data: {"a":1}\n\n{"b":2}\n'))).toEqual(['{"a":1}', '{"b":2}'])
  })

  it('flushes a final line without a trailing newline on end', () => {
    const read = createNdjsonReader()
    expect(read(enc.encode('{"a":1}'))).toEqual([])
    expect(read(undefined, true)).toEqual(['{"a":1}'])
  })
})
