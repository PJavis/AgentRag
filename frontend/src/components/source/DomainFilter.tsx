'use client'

import { useQuery } from '@tanstack/react-query'
import { Filter, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { ontologyApi, TaxonomyItem } from '@/lib/api/ontology'
import { DomainFilterValue } from '@/lib/types/api'
import { useTranslation } from '@/lib/hooks/use-translation'

const ANY_VALUE = '__any__'

const ONTOLOGY_QUERY_KEYS = {
  systems: ['ontology', 'systems'] as const,
  specialties: ['ontology', 'specialties'] as const,
}

interface DomainFilterProps {
  value: DomainFilterValue | null
  onChange: (value: DomainFilterValue | null) => void
  disabled?: boolean
}

export function DomainFilter({ value, onChange, disabled }: DomainFilterProps) {
  const { t } = useTranslation()

  const { data: systems = [] } = useQuery<TaxonomyItem[]>({
    queryKey: ONTOLOGY_QUERY_KEYS.systems,
    queryFn: ontologyApi.systems,
    staleTime: 60 * 60 * 1000, // 1h — taxonomy rarely changes
  })

  const { data: specialties = [] } = useQuery<TaxonomyItem[]>({
    queryKey: ONTOLOGY_QUERY_KEYS.specialties,
    queryFn: ontologyApi.specialties,
    staleTime: 60 * 60 * 1000,
  })

  const selectedSystem = value?.system ?? null
  const selectedSpecialty =
    value?.specialties && value.specialties.length > 0 ? value.specialties[0] : null

  const activeCount = (selectedSystem ? 1 : 0) + (selectedSpecialty ? 1 : 0)

  const updateSystem = (next: string) => {
    const nextSystem = next === ANY_VALUE ? null : next
    if (!nextSystem && !selectedSpecialty) {
      onChange(null)
      return
    }
    onChange({
      system: nextSystem ?? undefined,
      specialties: selectedSpecialty ? [selectedSpecialty] : undefined,
    })
  }

  const updateSpecialty = (next: string) => {
    const nextSpecialty = next === ANY_VALUE ? null : next
    if (!selectedSystem && !nextSpecialty) {
      onChange(null)
      return
    }
    onChange({
      system: selectedSystem ?? undefined,
      specialties: nextSpecialty ? [nextSpecialty] : undefined,
    })
  }

  const clear = () => onChange(null)

  const systemLabel =
    systems.find((s) => s.value === selectedSystem)?.label ?? null
  const specialtyLabel =
    specialties.find((s) => s.value === selectedSpecialty)?.label ?? null

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1.5 text-xs"
          disabled={disabled}
        >
          <Filter className="h-3.5 w-3.5" />
          <span>{t('chat.domainFilter') || 'Lĩnh vực'}</span>
          {activeCount > 0 && (
            <Badge variant="secondary" className="h-4 px-1 text-[10px]">
              {activeCount}
            </Badge>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 space-y-3">
        <div className="space-y-1.5">
          <Label className="text-xs">
            {t('chat.domainSystem') || 'Hệ cơ quan'}
          </Label>
          <Select
            value={selectedSystem ?? ANY_VALUE}
            onValueChange={updateSystem}
            disabled={disabled}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder={t('common.any') || 'Tất cả'} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ANY_VALUE} className="text-xs">
                {t('common.any') || 'Tất cả'}
              </SelectItem>
              {systems.map((s) => (
                <SelectItem key={s.value} value={s.value} className="text-xs">
                  {s.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-xs">
            {t('chat.domainSpecialty') || 'Chuyên khoa'}
          </Label>
          <Select
            value={selectedSpecialty ?? ANY_VALUE}
            onValueChange={updateSpecialty}
            disabled={disabled}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder={t('common.any') || 'Tất cả'} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ANY_VALUE} className="text-xs">
                {t('common.any') || 'Tất cả'}
              </SelectItem>
              {specialties.map((s) => (
                <SelectItem key={s.value} value={s.value} className="text-xs">
                  {s.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {(systemLabel || specialtyLabel) && (
          <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t">
            {systemLabel && (
              <Badge variant="outline" className="text-[10px]">
                {systemLabel}
              </Badge>
            )}
            {specialtyLabel && (
              <Badge variant="outline" className="text-[10px]">
                {specialtyLabel}
              </Badge>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-6 ml-auto gap-1 text-xs"
              onClick={clear}
            >
              <X className="h-3 w-3" />
              {t('common.clear') || 'Xóa'}
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}

export default DomainFilter
