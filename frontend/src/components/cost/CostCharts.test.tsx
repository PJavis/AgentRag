import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { CostCharts } from './CostCharts'

vi.mock('recharts', async (orig) => {
  const actual = await orig<typeof import('recharts')>()
  return { ...actual, ResponsiveContainer: ({ children }: { children: React.ReactNode }) =>
    <div style={{ width: 400, height: 200 }}>{children}</div> }
})

describe('CostCharts', () => {
  it('renders without crashing with empty data', () => {
    const { container } = render(<CostCharts perModel={{}} recent={[]} />)
    expect(container).toBeTruthy()
  })
})
