'use client'

import { useState } from 'react'
import { ThumbsUp, ThumbsDown } from 'lucide-react'
import apiClient from '@/lib/api/client'
import { Button } from '@/components/ui/button'

interface FeedbackButtonsProps {
  turnId: string
  sessionId?: string
  question?: string
  answer: string
  reasoningPath?: string
}

type Rating = 1 | -1 | null

export function FeedbackButtons({
  turnId,
  sessionId,
  question,
  answer,
  reasoningPath,
}: FeedbackButtonsProps) {
  const [rating, setRating] = useState<Rating>(null)
  const [sending, setSending] = useState(false)

  const submit = async (next: 1 | -1) => {
    if (sending) return
    const value = rating === next ? null : next
    setRating(value)
    if (value === null) return
    setSending(true)
    try {
      await apiClient.post('/chat/feedback', {
        turn_id: turnId,
        session_id: sessionId,
        rating: value,
        question,
        answer,
        reasoning_path: reasoningPath,
      })
    } catch {
      // revert visual on failure
      setRating(null)
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex items-center gap-1">
      <Button
        variant="ghost"
        size="icon"
        aria-label="Helpful"
        className={`h-6 w-6 ${rating === 1 ? 'text-green-600' : 'text-muted-foreground'}`}
        onClick={() => submit(1)}
        disabled={sending}
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        aria-label="Not helpful"
        className={`h-6 w-6 ${rating === -1 ? 'text-red-600' : 'text-muted-foreground'}`}
        onClick={() => submit(-1)}
        disabled={sending}
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}
