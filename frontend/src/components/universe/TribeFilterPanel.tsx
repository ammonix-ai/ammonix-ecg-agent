import { useState, useMemo } from 'react'
import { useVisualisationStore } from '@/stores/visualisationStore'
import { useAvailableTribes } from '@/stores/visualisationSelectors'
import { cn } from '@/design/cn'
import { X, Search } from 'lucide-react'

interface TribeFilterPanelProps {
  className?: string
}

interface TribeGroup {
  diagnosis: string
  tribes: { id: string; count: number }[]
  totalCount: number
}

export function TribeFilterPanel({ className }: TribeFilterPanelProps) {
  const points = useVisualisationStore((s) => s.points)
  const selectedTribes = useVisualisationStore((s) => s.selectedTribes)
  const showNoTribes = useVisualisationStore((s) => s.showNoTribes)
  const setShowTribeFilter = useVisualisationStore((s) => s.setShowTribeFilter)
  const toggleTribe = useVisualisationStore((s) => s.toggleTribe)
  const setTribes = useVisualisationStore((s) => s.setTribes)
  const setShowNoTribes = useVisualisationStore((s) => s.setShowNoTribes)
  const availableTribes = useAvailableTribes()

  const [searchTerm, setSearchTerm] = useState('')

  // Group tribes by most common primary diagnosis
  const tribeGroups = useMemo<TribeGroup[]>(() => {
    if (points.length === 0) return []

    const tribeStats = new Map<string, { diagCounts: Map<string, number>; total: number }>()

    for (const point of points) {
      const pt = point as typeof point & { tribes?: string[]; primaryDiagnosis?: string }
      if (!pt.tribes) continue
      for (const tribeId of pt.tribes) {
        let stats = tribeStats.get(tribeId)
        if (!stats) {
          stats = { diagCounts: new Map(), total: 0 }
          tribeStats.set(tribeId, stats)
        }
        stats.total++
        const diag = pt.primaryDiagnosis ?? 'Unknown'
        const dc = stats.diagCounts.get(diag) || 0
        stats.diagCounts.set(diag, dc + 1)
      }
    }

    const groups = new Map<string, { id: string; count: number }[]>()

    for (const tribeId of availableTribes) {
      const stats = tribeStats.get(tribeId)
      if (!stats) continue

      let topDiag = 'Unknown'
      let topCount = 0
      for (const [diag, count] of stats.diagCounts) {
        if (count > topCount) {
          topDiag = diag
          topCount = count
        }
      }

      let group = groups.get(topDiag)
      if (!group) {
        group = []
        groups.set(topDiag, group)
      }
      group.push({ id: tribeId, count: stats.total })
    }

    return Array.from(groups.entries())
      .map(([diagnosis, tribes]) => ({
        diagnosis,
        tribes: tribes.sort((a, b) => a.id.localeCompare(b.id)),
        totalCount: tribes.reduce((sum, t) => sum + t.count, 0),
      }))
      .sort((a, b) => a.diagnosis.localeCompare(b.diagnosis))
  }, [points, availableTribes])

  // Filter by search
  const filteredGroups = useMemo(() => {
    if (!searchTerm.trim()) return tribeGroups
    const q = searchTerm.toLowerCase()
    return tribeGroups
      .map((group) => ({
        ...group,
        tribes: group.tribes.filter(
          (t) => t.id.toLowerCase().includes(q) || group.diagnosis.toLowerCase().includes(q),
        ),
      }))
      .filter((g) => g.tribes.length > 0)
  }, [tribeGroups, searchTerm])

  const handleSelectAll = () => {
    setTribes(new Set(availableTribes))
    setShowNoTribes(false)
  }

  const handleClear = () => {
    setTribes(new Set())
    setShowNoTribes(false)
  }

  const handleNone = () => {
    setTribes(new Set())
    setShowNoTribes(true)
  }

  return (
    <div className={cn('bg-white rounded-xl border border-gray-200 shadow-lg', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-gray-900">Tribe Filter</h3>
          {selectedTribes.size > 0 && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium bg-brand-100 text-brand-700 rounded-full">
              {selectedTribes.size} / {availableTribes.length}
            </span>
          )}
        </div>
        <button
          onClick={() => setShowTribeFilter(false)}
          className="p-1 rounded hover:bg-gray-100 transition-colors"
        >
          <X className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      <div className="p-4 space-y-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
          <input
            type="text"
            placeholder="Search tribes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-8 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded hover:bg-gray-100"
            >
              <X className="w-3 h-3 text-gray-400" />
            </button>
          )}
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleSelectAll}
            className="px-2 py-1 text-xs font-medium rounded border border-gray-300 hover:bg-gray-50 transition-colors"
          >
            All
          </button>
          <button
            onClick={handleClear}
            className="px-2 py-1 text-xs font-medium rounded border border-gray-300 hover:bg-gray-50 transition-colors"
          >
            Clear
          </button>
          <button
            onClick={handleNone}
            className={cn(
              'px-2 py-1 text-xs font-medium rounded border transition-colors',
              showNoTribes
                ? 'border-amber-400 bg-amber-50 text-amber-700'
                : 'border-gray-300 hover:bg-gray-50',
            )}
          >
            None
          </button>
        </div>

        {/* Isolation mode warning */}
        {showNoTribes && (
          <div className="px-3 py-2 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-700">
            Isolation Mode: Only searched patients visible. Select a tribe or click &quot;All&quot; to exit.
          </div>
        )}

        {/* Tribe list */}
        <div className="max-h-[400px] overflow-y-auto space-y-3">
          {filteredGroups.map((group) => (
            <div key={group.diagnosis}>
              <div className="text-xs font-medium text-gray-500 mb-1">
                {group.diagnosis}{' '}
                <span className="text-gray-400">({group.totalCount})</span>
              </div>
              <div className="space-y-0.5">
                {group.tribes.map((tribe) => (
                  <label
                    key={tribe.id}
                    className={cn(
                      'flex items-center gap-2 px-2 py-1 rounded cursor-pointer hover:bg-gray-50 transition-colors',
                      selectedTribes.has(tribe.id) && 'bg-brand-50',
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={selectedTribes.has(tribe.id)}
                      onChange={() => toggleTribe(tribe.id)}
                      className="w-3.5 h-3.5 rounded border-gray-300 text-brand-600 focus:ring-brand-500"
                    />
                    <span className="text-xs text-gray-700 flex-1">{tribe.id}</span>
                    <span className="text-[10px] text-gray-400">{tribe.count}</span>
                  </label>
                ))}
              </div>
            </div>
          ))}

          {filteredGroups.length === 0 && (
            <div className="text-xs text-gray-400 text-center py-4">
              {searchTerm ? 'No tribes match your search' : 'No tribes available'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
