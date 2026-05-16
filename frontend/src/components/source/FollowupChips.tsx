'use client'

import { Lightbulb } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface FollowupChipsProps {
  suggestions: string[]
  onSelect: (q: string) => void
  disabled?: boolean
}

export function FollowupChips({ suggestions, onSelect, disabled }: FollowupChipsProps) {
  if (!suggestions || suggestions.length === 0) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground mr-1">
        <Lightbulb className="h-3 w-3" />
        Gợi ý
      </div>
      {suggestions.map((q, i) => (
        <Button
          key={`${i}-${q}`}
          variant="outline"
          size="sm"
          className="h-7 text-xs whitespace-normal text-left max-w-full"
          disabled={disabled}
          onClick={() => onSelect(q)}
        >
          {q}
        </Button>
      ))}
    </div>
  )
}

export default FollowupChips
