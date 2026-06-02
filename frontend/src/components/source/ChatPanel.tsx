'use client'

import { useState, useRef, useEffect, useId } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { Bot, Send, Loader2, FileText, Lightbulb, StickyNote, Clock, RefreshCcw, Paperclip, X, Square } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import {
  SourceChatMessage,
  SourceChatContextIndicator,
  BaseChatSession,
  DomainFilterValue,
  Citation,
} from '@/lib/types/api'
import { ModelSelector } from './ModelSelector'
import { DomainFilter } from './DomainFilter'
import { VerbosityToggle } from './VerbosityToggle'
import { TraceDialog } from './TraceDialog'
import { Network } from 'lucide-react'
import { ContextIndicator } from '@/components/common/ContextIndicator'
import { SessionManager } from '@/components/source/SessionManager'
import { MessageActions } from '@/components/source/MessageActions'
import { FeedbackButtons } from '@/components/source/FeedbackButtons'
import { FollowupChips } from '@/components/source/FollowupChips'
import { InlineImageCitation } from '@/components/source/InlineImageCitation'
import {
  convertReferencesToCompactMarkdown,
  createCompactReferenceLinkComponent,
  createCompactReferenceHoverComponent,
} from '@/lib/utils/source-references'
import { useModalManager } from '@/lib/hooks/use-modal-manager'
import { toast } from 'sonner'
import { useTranslation } from '@/lib/hooks/use-translation'
import type { StreamingStep } from '@/lib/hooks/useNotebookChat'

interface NotebookContextStats {
  sourcesInsights: number
  sourcesFull: number
  notesCount: number
  tokenCount?: number
  charCount?: number
}

interface ChatPanelProps {
  messages: SourceChatMessage[]
  isStreaming: boolean
  contextIndicators: SourceChatContextIndicator | null
  onSendMessage: (
    message: string,
    modelOverride?: string,
    domainFilter?: DomainFilterValue | null,
    verbosity?: 'concise' | 'detailed' | null
  ) => void
  onRegenerateMessage?: (
    assistantMessageId: string,
    domainFilter?: DomainFilterValue | null,
    verbosity?: 'concise' | 'detailed' | null
  ) => void
  onSendImageMessage?: (message: string, file: File) => void
  modelOverride?: string
  onModelChange?: (model?: string) => void
  // Session management props
  sessions?: BaseChatSession[]
  currentSessionId?: string | null
  onCreateSession?: (title: string) => void
  onSelectSession?: (sessionId: string) => void
  onDeleteSession?: (sessionId: string) => void
  onUpdateSession?: (sessionId: string, title: string) => void
  loadingSessions?: boolean
  // Generic props for reusability
  title?: string
  contextType?: 'source' | 'notebook'
  // Notebook context stats (for notebook chat)
  notebookContextStats?: NotebookContextStats
  // Notebook ID for saving notes
  notebookId?: string
  // Live streaming step for inline flow indicator (mirrors backend status steps)
  streamingStep?: StreamingStep
  // Active tool name when streamingStep === 'tool'
  streamingTool?: string | null
  // Cancel in-flight streaming (renders Stop button when present)
  onCancelStreaming?: () => void
  // Document-aware starter suggestions for the empty state (owned by parent query)
  dynamicStarters?: string[]
  startersLoading?: boolean
}

export function ChatPanel({
  messages,
  isStreaming,
  contextIndicators,
  onSendMessage,
  onRegenerateMessage,
  onSendImageMessage,
  modelOverride,
  onModelChange,
  sessions = [],
  currentSessionId,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
  onUpdateSession,
  loadingSessions = false,
  title,
  contextType = 'source',
  notebookContextStats,
  notebookId,
  streamingStep = 'idle',
  streamingTool,
  onCancelStreaming,
  dynamicStarters = [],
  startersLoading = false
}: ChatPanelProps) {
  const { t } = useTranslation()
  const chatInputId = useId()
  const [input, setInput] = useState('')
  const [sessionManagerOpen, setSessionManagerOpen] = useState(false)
  const [domainFilter, setDomainFilter] = useState<DomainFilterValue | null>(null)
  const [verbosity, setVerbosity] = useState<'concise' | 'detailed' | null>(null)
  const [attachedImage, setAttachedImage] = useState<File | null>(null)
  const attachedPreview = attachedImage ? URL.createObjectURL(attachedImage) : null
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [traceMessage, setTraceMessage] = useState<SourceChatMessage | null>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { openModal } = useModalManager()

  const handleReferenceClick = (type: string, id: string) => {
    const modalType = type === 'source_insight' ? 'insight' : type as 'source' | 'note' | 'insight'

    try {
      openModal(modalType, id)
      // Note: The modal system uses URL parameters and doesn't throw errors for missing items.
      // The modal component itself will handle displaying "not found" states.
      // This try-catch is here for future enhancements or unexpected errors.
    } catch {
      toast.error(t('common.noResults'))
    }
  }

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (isStreaming) return
    // Image-attached path overrides text-only.
    if (attachedImage && onSendImageMessage) {
      onSendImageMessage(input.trim(), attachedImage)
      setInput('')
      setAttachedImage(null)
      return
    }
    if (input.trim()) {
      onSendMessage(
        input.trim(),
        modelOverride,
        contextType === 'notebook' ? domainFilter : undefined,
        verbosity,
      )
      setInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Enter alone → send. Shift+Enter → newline. IME composition → ignore.
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      handleSend()
    }
  }

  const keyHint = 'Enter'

  // The last AI message is the active streaming target while isStreaming. We use
  // its id to (a) render an inline ThinkingIndicator when it has no content yet,
  // and (b) suppress its action row + follow-ups until the stream finishes — so
  // a half-written answer never shows save/regenerate/feedback or a duplicate
  // "Writing…" indicator beneath it.
  const lastMessage = messages[messages.length - 1]
  const streamingMsgId =
    isStreaming && lastMessage?.type === 'ai' ? lastMessage.id : null

  // Fixed universal starters (instant, client-side). `q` is the prompt sent.
  const FIXED_STARTERS: { label: string; q: string }[] = [
    { label: '📋 Tóm tắt tài liệu', q: 'Tóm tắt chi tiết tài liệu này' },
    { label: '🔍 Các điểm chính', q: 'Liệt kê các điểm chính trong tài liệu' },
  ]
  const sendStarter = (q: string) =>
    onSendMessage(
      q,
      modelOverride,
      contextType === 'notebook' ? domainFilter : undefined,
      verbosity,
    )

  // Per-answer quick actions — one tap re-asks with a refined intent (new turn).
  const QUICK_ACTIONS: { label: string; q: string }[] = [
    { label: '💡 Giải thích đơn giản', q: 'Giải thích lại câu trả lời trên một cách đơn giản, dễ hiểu hơn.' },
    { label: '➕ Chi tiết hơn', q: 'Trình bày chi tiết hơn, đầy đủ hơn về nội dung trên.' },
    { label: '🔑 Điểm chính', q: 'Liệt kê các điểm chính của nội dung trên dưới dạng gạch đầu dòng.' },
    { label: '📌 Cho ví dụ', q: 'Cho ví dụ minh hoạ cụ thể cho nội dung trên.' },
  ]
  // Always-on intent buttons near the input (available mid-conversation).
  const INTENT_BUTTONS: { label: string; q: string }[] = [
    { label: 'Tóm tắt', q: 'Tóm tắt chi tiết tài liệu' },
    { label: 'Giải thích', q: 'Giải thích chi tiết các khái niệm chính trong tài liệu' },
    { label: 'Điểm chính', q: 'Liệt kê các điểm chính trong tài liệu' },
    { label: 'So sánh', q: 'So sánh và đối chiếu các nội dung/quan điểm chính trong tài liệu' },
  ]

  return (
    <>
    <div className="relative flex flex-col h-full flex-1 overflow-hidden bg-background">
      {/* Sessions: floating top-right */}
      {onSelectSession && onCreateSession && onDeleteSession && (
        <div className="absolute top-3 right-3 z-10">
          <Dialog open={sessionManagerOpen} onOpenChange={setSessionManagerOpen}>
            <Button
              variant="ghost"
              size="sm"
              className="gap-2 h-8"
              onClick={() => setSessionManagerOpen(true)}
              disabled={loadingSessions}
              title={t('chat.sessions')}
            >
              <Clock className="h-4 w-4" />
              <span className="text-xs hidden sm:inline">{t('chat.sessions')}</span>
            </Button>
            <DialogContent className="sm:max-w-[420px] p-0 overflow-hidden">
              <DialogTitle className="sr-only">{t('chat.sessionsTitle')}</DialogTitle>
              <SessionManager
                sessions={sessions}
                currentSessionId={currentSessionId ?? null}
                onCreateSession={(title) => onCreateSession?.(title)}
                onSelectSession={(sessionId) => {
                  onSelectSession(sessionId)
                  setSessionManagerOpen(false)
                }}
                onUpdateSession={(sessionId, title) => onUpdateSession?.(sessionId, title)}
                onDeleteSession={(sessionId) => onDeleteSession?.(sessionId)}
                loadingSessions={loadingSessions}
              />
            </DialogContent>
          </Dialog>
        </div>
      )}

      <ScrollArea className="flex-1 min-h-0" ref={scrollAreaRef}>
        <div className="max-w-3xl mx-auto w-full px-4 sm:px-6 py-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center min-h-[60vh] gap-6">
              <div className="flex flex-col items-center">
                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                  <Bot className="h-6 w-6 text-primary" />
                </div>
                <h2 className="text-xl font-semibold mb-2">
                  {title || t('chat.startConversation').replace('{type}', contextType === 'source' ? t('navigation.sources') : t('common.notebook'))}
                </h2>
                <p className="text-sm text-muted-foreground">{t('chat.askQuestions')}</p>
              </div>
              {!isStreaming && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl" data-testid="chat-starters">
                  {FIXED_STARTERS.map((chip) => (
                    <button
                      key={chip.label}
                      type="button"
                      onClick={() => sendStarter(chip.q)}
                      className="text-left text-sm px-4 py-3 rounded-xl border bg-muted/40 hover:bg-accent transition-colors"
                    >
                      {chip.label}
                    </button>
                  ))}
                  {startersLoading
                    ? [0, 1, 2].map((i) => (
                        <div
                          key={`sk-${i}`}
                          data-testid="starter-skeleton"
                          className="h-[46px] rounded-xl border bg-muted/40 animate-pulse"
                        />
                      ))
                    : dynamicStarters.map((q) => (
                        <button
                          key={q}
                          type="button"
                          onClick={() => sendStarter(q)}
                          className="text-left text-sm px-4 py-3 rounded-xl border bg-muted/40 hover:bg-accent transition-colors"
                        >
                          💡 {q}
                        </button>
                      ))}
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-8">
              {messages.map((message) => (
                <div key={message.id}>
                  {message.type === 'human' ? (
                    <div className="flex justify-end">
                      <div className="max-w-[85%] rounded-2xl bg-muted px-4 py-2.5">
                        <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>
                      </div>
                    </div>
                  ) : (
                    (() => {
                    // Is THIS bubble the one currently being streamed into?
                    const isActiveStream = message.id === streamingMsgId
                    return (
                    <div className="group">
                      {isActiveStream && !message.content.trim() ? (
                        <ThinkingIndicator
                          step={streamingStep}
                          tool={streamingTool}
                          onCancel={onCancelStreaming}
                        />
                      ) : (
                        <AIMessageContent
                          content={message.content}
                          citations={(message.citations as Citation[] | undefined) || []}
                          onReferenceClick={handleReferenceClick}
                        />
                      )}
                      {/* Action row + follow-ups: hidden on the actively-streaming
                          bubble (no premature save/regenerate/feedback) and on any
                          empty bubble (so a content-less placeholder shows nothing). */}
                      {!isActiveStream && message.content.trim() && (
                      <div className="flex items-center justify-between gap-2 mt-1 opacity-70 group-hover:opacity-100 transition-opacity">
                        <div className="flex items-center gap-1 flex-wrap">
                          <MessageActions
                            content={message.content}
                            notebookId={notebookId}
                          />
                          {readingMinutes(message.content) >= 2 && (
                            <Badge variant="outline" className="text-[10px] h-5 gap-1" title="Thời gian đọc ước tính">
                              <Clock className="h-3 w-3" />
                              {readingMinutes(message.content)} phút đọc
                            </Badge>
                          )}
                          {(message as { reasoning_path?: string }).reasoning_path && (
                            <Badge
                              variant="outline"
                              className="text-[10px] h-5 font-mono"
                              title="Reasoning path used for this answer"
                            >
                              {(message as { reasoning_path?: string }).reasoning_path}
                            </Badge>
                          )}
                          {(message.tool_trace?.length || message.timings_ms) && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 gap-1 text-xs"
                              onClick={() => setTraceMessage(message)}
                              title={t('chat.viewTrace') || 'View reasoning trace'}
                            >
                              <Network className="h-3.5 w-3.5" />
                              <span>{t('chat.trace') || 'Trace'}</span>
                            </Button>
                          )}
                          {onRegenerateMessage && !message.id.startsWith('temp-') && !message.id.startsWith('local-') && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7"
                              disabled={isStreaming}
                              onClick={() =>
                                onRegenerateMessage(
                                  message.id,
                                  contextType === 'notebook' ? domainFilter : undefined,
                                  verbosity,
                                )
                              }
                              title="Sinh lại câu trả lời"
                              aria-label="Sinh lại câu trả lời"
                            >
                              <RefreshCcw className="h-3.5 w-3.5" />
                            </Button>
                          )}
                        </div>
                        <FeedbackButtons
                          turnId={message.id}
                          sessionId={currentSessionId ?? undefined}
                          answer={message.content}
                        />
                      </div>
                      )}
                      {!isActiveStream && message.content.trim() && (
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {QUICK_ACTIONS.map((qa) => (
                            <button
                              key={qa.label}
                              type="button"
                              disabled={isStreaming}
                              onClick={() => sendStarter(qa.q)}
                              className="text-[11px] px-2.5 py-1 rounded-full border bg-muted/30 hover:bg-accent disabled:opacity-50 transition-colors"
                            >
                              {qa.label}
                            </button>
                          ))}
                        </div>
                      )}
                      {!isActiveStream && message.follow_ups && message.follow_ups.length > 0 && (
                        <div className="mt-3">
                          <FollowupChips
                            suggestions={message.follow_ups}
                            onSelect={(q) =>
                              onSendMessage(
                                q,
                                modelOverride,
                                contextType === 'notebook' ? domainFilter : undefined,
                                verbosity,
                              )
                            }
                            disabled={isStreaming}
                          />
                        </div>
                      )}
                    </div>
                    )
                    })()
                  )}
                </div>
              ))}
              {/* Fallback flow indicator for streaming flows that don't add an
                  inline AI placeholder (e.g. regenerate). Never doubles with the
                  inline indicator above. */}
              {isStreaming && !streamingMsgId && (
                <ThinkingIndicator
                  step={streamingStep}
                  tool={streamingTool}
                  onCancel={onCancelStreaming}
                />
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Context Indicators */}
      {contextIndicators && (
        <div className="max-w-3xl mx-auto w-full px-4 sm:px-6 py-2">
          <div className="flex flex-wrap gap-2 text-xs">
            {contextIndicators.sources?.length > 0 && (
              <Badge variant="outline" className="gap-1">
                <FileText className="h-3 w-3" />
                {contextIndicators.sources.length} {t('navigation.sources')}
              </Badge>
            )}
            {contextIndicators.insights?.length > 0 && (
              <Badge variant="outline" className="gap-1">
                <Lightbulb className="h-3 w-3" />
                {contextIndicators.insights.length} {contextIndicators.insights.length === 1 ? t('common.insight') : t('common.insights')}
              </Badge>
            )}
            {contextIndicators.notes?.length > 0 && (
              <Badge variant="outline" className="gap-1">
                <StickyNote className="h-3 w-3" />
                {contextIndicators.notes.length} {contextIndicators.notes.length === 1 ? t('common.note') : t('common.notes')}
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Notebook Context Indicator */}
      {notebookContextStats && (
        <div className="max-w-3xl mx-auto w-full px-4 sm:px-6">
          <ContextIndicator
            sourcesInsights={notebookContextStats.sourcesInsights}
            sourcesFull={notebookContextStats.sourcesFull}
            notesCount={notebookContextStats.notesCount}
            tokenCount={notebookContextStats.tokenCount}
            charCount={notebookContextStats.charCount}
          />
        </div>
      )}

      {/* Input Area: floating pill, centered */}
      <div className="flex-shrink-0 px-4 sm:px-6 pb-4 pt-2">
        <div className="max-w-3xl mx-auto w-full space-y-3">
          {/* Always-on intent buttons — quick ways to ask, shown once a chat is
              underway (the empty state already shows the bigger starter cards). */}
          {messages.length > 0 && !attachedImage && (
            <div className="flex flex-wrap gap-1.5">
              {INTENT_BUTTONS.map((b) => (
                <button
                  key={b.label}
                  type="button"
                  disabled={isStreaming}
                  onClick={() => sendStarter(b.q)}
                  className="text-[11px] px-2.5 py-1 rounded-full border bg-muted/30 hover:bg-accent disabled:opacity-50 transition-colors"
                >
                  {b.label}
                </button>
              ))}
            </div>
          )}
          {/* Model selector + Verbosity + Domain filter */}
          {(onModelChange || contextType === 'notebook') && (
            <div className="flex items-center justify-between gap-2 text-xs">
              {onModelChange ? (
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">{t('chat.model')}</span>
                  <ModelSelector
                    currentModel={modelOverride}
                    onModelChange={onModelChange}
                    disabled={isStreaming}
                  />
                </div>
              ) : (
                <span />
              )}
              <div className="flex items-center gap-2">
                <VerbosityToggle value={verbosity} onChange={setVerbosity} disabled={isStreaming} />
                {contextType === 'notebook' && (
                  <DomainFilter
                    value={domainFilter}
                    onChange={setDomainFilter}
                    disabled={isStreaming}
                  />
                )}
              </div>
            </div>
          )}

          {attachedPreview && (
            <div className="flex items-center gap-2 px-2 py-1 border rounded-xl bg-muted/30">
              <img
                src={attachedPreview}
                alt="attached"
                className="h-12 w-12 object-cover rounded"
              />
              <span className="text-xs flex-1 truncate text-muted-foreground">
                {attachedImage?.name} · {((attachedImage?.size ?? 0) / 1024).toFixed(0)} KB
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={() => setAttachedImage(null)}
                title="Bỏ ảnh đính kèm"
              >
                <X className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}

          {/* Pill input row */}
          <div className="flex gap-2 items-end min-w-0 rounded-2xl border bg-background shadow-sm px-2 py-1.5 focus-within:ring-1 focus-within:ring-ring transition-shadow">
            {onSendImageMessage && (
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) setAttachedImage(f)
                    e.target.value = ''
                  }}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 flex-shrink-0 rounded-full"
                  disabled={isStreaming}
                  onClick={() => fileInputRef.current?.click()}
                  title="Đính kèm ảnh"
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
              </>
            )}
            <Textarea
              id={chatInputId}
              name="chat-message"
              autoComplete="off"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={(() => {
                if (attachedImage) return 'Hỏi gì về ảnh này? (Enter để gửi)'
                const main = t('chat.sendPlaceholder') || 'Ask anything…'
                const press = t('chat.pressToSend') || 'Press {key} to send'
                const hint = (typeof press === 'string' ? press : 'Press {key} to send').replace('{key}', keyHint)
                return `${main} (${hint})`
              })()}
              disabled={isStreaming}
              className="flex-1 min-h-[40px] max-h-[160px] resize-none py-2 px-2 min-w-0 border-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent"
              rows={1}
            />
            {isStreaming && onCancelStreaming ? (
              <Button
                onClick={onCancelStreaming}
                size="icon"
                variant="destructive"
                className="h-9 w-9 flex-shrink-0 rounded-full"
                title="Stop generating"
                aria-label="Stop generating"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </Button>
            ) : (
              <Button
                onClick={handleSend}
                disabled={(!input.trim() && !attachedImage) || isStreaming}
                size="icon"
                className="h-9 w-9 flex-shrink-0 rounded-full"
              >
                {isStreaming ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
    <TraceDialog
      open={!!traceMessage}
      onOpenChange={(v) => { if (!v) setTraceMessage(null) }}
      message={traceMessage as never}
    />
    </>
  )
}

// Helper component to render AI messages with hover-card citations + inline images
function AIMessageContent({
  content,
  citations,
  onReferenceClick,
}: {
  content: string
  citations: Citation[]
  onReferenceClick: (type: string, id: string) => void
}) {
  const { t } = useTranslation()
  // Never render an empty prose container — an AI bubble with no content (e.g. a
  // completion that streamed no tokens and skipped reconciliation) would otherwise
  // show up as a stray empty bar. The active-stream placeholder is handled by the
  // ThinkingIndicator branch upstream; here we just render nothing.
  if (!content.trim()) return null
  let markdownWithCompactRefs = convertReferencesToCompactMarkdown(content, t('common.references'))
  // Turn the model's inline [n] markers into reference links so they render as
  // clickable + hover-to-read citation badges (HoverComponent resolves [n] →
  // citations[n-1]). Skip [n] already followed by '(' (already a link).
  if (citations && citations.length > 0) {
    markdownWithCompactRefs = markdownWithCompactRefs.replace(
      /\[(\d{1,3})\](?!\()/g,
      (m, n) => {
        const num = parseInt(n, 10)
        return num >= 1 && num <= citations.length ? `[${num}](#ref-c-${num})` : m
      },
    )
  }
  // Prefer hover when citations are present; fall back to clickable when not.
  const LinkComponent = citations && citations.length > 0
    ? createCompactReferenceHoverComponent(citations, onReferenceClick)
    : createCompactReferenceLinkComponent(onReferenceClick)
  const imageCitations = (citations || []).filter((c) => c.segment_type === 'image' && c.image_url)
  return (
    <div className="prose prose-sm prose-neutral dark:prose-invert max-w-none break-words prose-headings:font-semibold prose-a:text-blue-600 prose-a:break-all prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-p:mb-4 prose-p:leading-7 prose-li:mb-2">
      {imageCitations.map((c, i) => (
        <InlineImageCitation key={`img-${i}-${c.content_hash ?? i}`} citation={c} />
      ))}
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: LinkComponent,
          p: ({ children }) => <p className="mb-4">{children}</p>,
          h1: ({ children }) => <h1 className="text-2xl font-bold mb-4 mt-6 pb-1 border-b border-border">{children}</h1>,
          h2: ({ children }) => <h2 className="text-xl font-semibold mb-3 mt-6">{children}</h2>,
          h3: ({ children }) => <h3 className="text-lg font-semibold mb-2 mt-4">{children}</h3>,
          h4: ({ children }) => <h4 className="mb-2 mt-4">{children}</h4>,
          h5: ({ children }) => <h5 className="mb-2 mt-3">{children}</h5>,
          h6: ({ children }) => <h6 className="mb-2 mt-3">{children}</h6>,
          li: ({ children }) => <li className="mb-1">{children}</li>,
          ul: ({ children }) => <ul className="mb-4 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="mb-4 space-y-1">{children}</ol>,
          strong: ({ children }) => (
            <strong className="font-semibold text-amber-900 dark:text-amber-200 bg-amber-100/60 dark:bg-amber-900/30 px-1 rounded">
              {children}
            </strong>
          ),
          em: ({ children }) => <em className="italic text-foreground/90">{children}</em>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-amber-400 bg-amber-50/60 dark:bg-amber-950/30 pl-4 pr-3 py-2 my-3 rounded-r">
              {children}
            </blockquote>
          ),
          code: ({ children, ...props }) => {
            const isInline = !(props as { className?: string }).className
            return isInline
              ? <code className="bg-muted px-1 py-0.5 rounded text-[0.9em]">{children}</code>
              : <code {...props}>{children}</code>
          },
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto">
              <table className="min-w-full border-collapse border border-border">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
          tbody: ({ children }) => <tbody>{children}</tbody>,
          tr: ({ children }) => <tr className="border-b border-border">{children}</tr>,
          th: ({ children }) => <th className="border border-border px-3 py-2 text-left font-semibold">{children}</th>,
          td: ({ children }) => <td className="border border-border px-3 py-2">{children}</td>,
        }}
      >
        {markdownWithCompactRefs}
      </ReactMarkdown>
    </div>
  )
}

// Animated flow indicator shown while the assistant prepares a reply. `step`
// mirrors the backend `event: status` pipeline steps, so the user sees the real
// agent flow (classifying → retrieving → running a tool → answering) instead of
// a dumb spinner. `tool` carries the active tool name when step === 'tool'.
function ThinkingIndicator({
  step,
  tool,
  onCancel,
}: {
  step: StreamingStep
  tool?: string | null
  onCancel?: () => void
}) {
  // Elapsed timer so a long detailed answer doesn't feel frozen.
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const t0 = Date.now()
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000)
    return () => clearInterval(id)
  }, [])
  const label = (() => {
    switch (step) {
      case 'connecting':
        return 'Connecting…'
      case 'chitchat':
        return 'Replying…'
      case 'classify':
        return 'Understanding your question…'
      case 'structured_reasoning':
        return 'Analyzing structured data…'
      case 'retrieve':
        return 'Searching documents…'
      case 'decide':
        return 'Planning next step…'
      case 'tool': {
        const pretty = tool ? prettyToolName(tool) : ''
        return pretty ? `Running tool: ${pretty}…` : 'Running tool…'
      }
      case 'answer':
        return 'Composing answer…'
      case 'writing':
        return 'Writing…'
      case 'finalizing':
        return 'Finalizing…'
      default:
        return 'Thinking…'
    }
  })()
  return (
    <div className="flex items-center gap-3 text-muted-foreground" role="status" aria-live="polite">
      <div className="flex items-center gap-1" aria-hidden>
        <span className="h-1.5 w-1.5 rounded-full bg-current animate-bounce [animation-delay:-0.3s]" />
        <span className="h-1.5 w-1.5 rounded-full bg-current animate-bounce [animation-delay:-0.15s]" />
        <span className="h-1.5 w-1.5 rounded-full bg-current animate-bounce" />
      </div>
      <span className="text-xs transition-opacity duration-150">{label}</span>
      {elapsed >= 3 && <span className="text-[10px] tabular-nums opacity-60">{elapsed}s</span>}
      {onCancel && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs gap-1"
          onClick={onCancel}
          title="Stop generating"
        >
          <Square className="h-3 w-3 fill-current" />
          Stop
        </Button>
      )}
    </div>
  )
}

// Estimated reading time in minutes (~200 words/min) for an answer's markdown.
function readingMinutes(text: string): number {
  const words = (text || '').trim().split(/\s+/).filter(Boolean).length
  return Math.max(1, Math.round(words / 200))
}

// Map raw backend tool identifiers to human-friendly names for the flow label.
function prettyToolName(tool: string): string {
  const map: Record<string, string> = {
    search_hybrid_kg: 'hybrid search',
    search_hybrid: 'hybrid search',
    search_vector: 'vector search',
    search_keyword: 'keyword search',
    search_sql: 'database query',
    sql: 'database query',
  }
  return map[tool] || tool.replace(/_/g, ' ')
}
