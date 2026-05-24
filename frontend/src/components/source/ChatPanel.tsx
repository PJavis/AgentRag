'use client'

import { useState, useRef, useEffect, useId } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { Bot, User, Send, Loader2, FileText, Lightbulb, StickyNote, Clock, RefreshCcw, Paperclip, X } from 'lucide-react'
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
  notebookId
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

  return (
    <>
    <Card className="flex flex-col h-full flex-1 overflow-hidden">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5" />
            {title || (contextType === 'source' ? t('chat.chatWith').replace('{name}', t('navigation.sources')) : t('chat.chatWith').replace('{name}', t('common.notebook')))}
          </CardTitle>
          {onSelectSession && onCreateSession && onDeleteSession && (
            <Dialog open={sessionManagerOpen} onOpenChange={setSessionManagerOpen}>
              <Button
                variant="ghost"
                size="sm"
                className="gap-2"
                onClick={() => setSessionManagerOpen(true)}
                disabled={loadingSessions}
              >
                <Clock className="h-4 w-4" />
                <span className="text-xs">{t('chat.sessions')}</span>
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
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 p-0">
        <ScrollArea className="flex-1 min-h-0 px-4" ref={scrollAreaRef}>
          <div className="space-y-4 py-4">
            {messages.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-sm">
                  {t('chat.startConversation').replace('{type}', contextType === 'source' ? t('navigation.sources') : t('common.notebook'))}
                </p>
                <p className="text-xs mt-2">{t('chat.askQuestions')}</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${
                    message.type === 'human' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  {message.type === 'ai' && (
                    <div className="flex-shrink-0">
                      <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                        <Bot className="h-4 w-4" />
                      </div>
                    </div>
                  )}
                  <div className="flex flex-col gap-2 max-w-[80%]">
                    <div
                      className={`rounded-lg px-4 py-2 ${
                        message.type === 'human'
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted'
                      }`}
                    >
                      {message.type === 'ai' ? (
                        <AIMessageContent
                          content={message.content}
                          citations={(message.citations as Citation[] | undefined) || []}
                          onReferenceClick={handleReferenceClick}
                        />
                      ) : (
                        <p className="text-sm break-all">{message.content}</p>
                      )}
                    </div>
                    {message.type === 'ai' && (
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-1">
                          <MessageActions
                            content={message.content}
                            notebookId={notebookId}
                          />
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
                    {message.type === 'ai' && message.follow_ups && message.follow_ups.length > 0 && (
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
                    )}
                  </div>
                  {message.type === 'human' && (
                    <div className="flex-shrink-0">
                      <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center">
                        <User className="h-4 w-4 text-primary-foreground" />
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
            {isStreaming && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4" />
                  </div>
                </div>
                <div className="rounded-lg px-4 py-2 bg-muted">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Context Indicators */}
        {contextIndicators && (
          <div className="border-t px-4 py-2">
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
          <ContextIndicator
            sourcesInsights={notebookContextStats.sourcesInsights}
            sourcesFull={notebookContextStats.sourcesFull}
            notesCount={notebookContextStats.notesCount}
            tokenCount={notebookContextStats.tokenCount}
            charCount={notebookContextStats.charCount}
          />
        )}

        {/* Input Area */}
        <div className="flex-shrink-0 p-4 space-y-3 border-t">
          {/* Quick-start chips — show when input is empty and no messages yet */}
          {!input.trim() && messages.length === 0 && !isStreaming && (
            <div className="flex flex-wrap gap-2">
              {[
                { label: '📋 Tóm tắt tài liệu', q: 'Tóm tắt chi tiết tài liệu này' },
                { label: '🔍 Các điểm chính', q: 'Liệt kê các điểm chính trong tài liệu' },
                { label: '❓ Câu hỏi thường gặp', q: 'Liệt kê các câu hỏi thường gặp về tài liệu này' },
              ].map((chip) => (
                <button
                  key={chip.label}
                  type="button"
                  className="text-xs px-3 py-1.5 rounded-full border bg-muted/40 hover:bg-accent transition-colors"
                  onClick={() =>
                    onSendMessage(
                      chip.q,
                      modelOverride,
                      contextType === 'notebook' ? domainFilter : undefined,
                      verbosity,
                    )
                  }
                >
                  {chip.label}
                </button>
              ))}
            </div>
          )}
          {/* Model selector + Domain filter (notebook only) */}
          {(onModelChange || contextType === 'notebook') && (
            <div className="flex items-center justify-between gap-2">
              {onModelChange ? (
                <>
                  <span className="text-xs text-muted-foreground">{t('chat.model')}</span>
                  <ModelSelector
                    currentModel={modelOverride}
                    onModelChange={onModelChange}
                    disabled={isStreaming}
                  />
                </>
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
            <div className="flex items-center gap-2 px-2 py-1 border rounded bg-muted/30">
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

          <div className="flex gap-2 items-end min-w-0">
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
                    e.target.value = '' // reset so same file can be re-selected
                  }}
                />
                <Button
                  variant="outline"
                  size="icon"
                  className="h-[40px] w-[40px] flex-shrink-0"
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
              className="flex-1 min-h-[40px] max-h-[100px] resize-none py-2 px-3 min-w-0"
              rows={1}
            />
            <Button
              onClick={handleSend}
              disabled={(!input.trim() && !attachedImage) || isStreaming}
              size="icon"
              className="h-[40px] w-[40px] flex-shrink-0"
            >
              {isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
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
  const markdownWithCompactRefs = convertReferencesToCompactMarkdown(content, t('common.references'))
  // Prefer hover when citations are present; fall back to clickable when not.
  const LinkComponent = citations && citations.length > 0
    ? createCompactReferenceHoverComponent(citations)
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
          h1: ({ children }) => <h1 className="mb-4 mt-6">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-3 mt-5">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-3 mt-4">{children}</h3>,
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
