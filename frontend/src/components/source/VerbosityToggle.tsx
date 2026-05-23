'use client'

import { AlignLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'

type Verbosity = 'concise' | 'detailed' | null

interface VerbosityToggleProps {
  value: Verbosity
  onChange: (value: Verbosity) => void
  disabled?: boolean
}

const LABEL: Record<NonNullable<Verbosity>, string> = {
  concise: 'Ngắn gọn',
  detailed: 'Chi tiết',
}

export function VerbosityToggle({ value, onChange, disabled }: VerbosityToggleProps) {
  const active = value !== null
  const display = active ? LABEL[value as NonNullable<Verbosity>] : 'Độ dài'

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant={active ? 'secondary' : 'outline'}
          size="sm"
          className="gap-1 h-8"
          disabled={disabled}
        >
          <AlignLeft className="h-3.5 w-3.5" />
          <span className="text-xs">{display}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-56 p-2" align="end">
        <div className="flex flex-col gap-1">
          <button
            type="button"
            onClick={() => onChange(null)}
            className={`text-left px-2 py-1.5 text-sm rounded hover:bg-accent ${value === null ? 'bg-accent font-medium' : ''}`}
          >
            <div>Tự động</div>
            <div className="text-xs text-muted-foreground">Phát hiện theo từ khoá ("chi tiết", "dài hơn", …)</div>
          </button>
          <button
            type="button"
            onClick={() => onChange('concise')}
            className={`text-left px-2 py-1.5 text-sm rounded hover:bg-accent ${value === 'concise' ? 'bg-accent font-medium' : ''}`}
          >
            <div>Ngắn gọn</div>
            <div className="text-xs text-muted-foreground">Trả lời trực tiếp, không lan man</div>
          </button>
          <button
            type="button"
            onClick={() => onChange('detailed')}
            className={`text-left px-2 py-1.5 text-sm rounded hover:bg-accent ${value === 'detailed' ? 'bg-accent font-medium' : ''}`}
          >
            <div>Chi tiết</div>
            <div className="text-xs text-muted-foreground">Mở rộng, nhiều đoạn, đi sâu từng ý</div>
          </button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
