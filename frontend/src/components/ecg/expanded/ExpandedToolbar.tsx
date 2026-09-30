import { Move, Ruler, Pencil, Compass, RotateCcw, Heart, Crosshair } from 'lucide-react'

export type ExpandedToolMode = 'navigate' | 'caliper' | 'annotate' | 'ruler' | 'hr-calc'

interface ExpandedToolbarProps {
  activeTool: ExpandedToolMode
  onToolChange: (tool: ExpandedToolMode) => void
  onReset?: () => void
  caliperCount?: number
  annotationCount?: number
  rulerCount?: number
  precisionMode?: boolean
  onPrecisionModeChange?: (enabled: boolean) => void
  disabled?: boolean
  className?: string
}

const TOOLS: { key: ExpandedToolMode; icon: typeof Move; label: string; shortcut: string; countKey?: string }[] = [
  { key: 'navigate', icon: Move, label: 'Navigate', shortcut: 'V' },
  { key: 'caliper', icon: Ruler, label: 'Caliper', shortcut: 'C', countKey: 'caliperCount' },
  { key: 'annotate', icon: Pencil, label: 'Draw', shortcut: 'A', countKey: 'annotationCount' },
  { key: 'ruler', icon: Compass, label: 'Ruler', shortcut: 'R', countKey: 'rulerCount' },
  { key: 'hr-calc', icon: Heart, label: 'HR Calc', shortcut: 'H' },
]

/**
 * ExpandedToolbar — Tool selection for expanded lead view
 */
export function ExpandedToolbar({
  activeTool, onToolChange, onReset,
  caliperCount = 0, annotationCount = 0, rulerCount = 0,
  precisionMode = false, onPrecisionModeChange,
  disabled = false, className,
}: ExpandedToolbarProps) {
  const counts: Record<string, number> = { caliperCount, annotationCount, rulerCount }
  const total = caliperCount + annotationCount + rulerCount

  return (
    <div className={`flex items-center gap-1 p-1 bg-gray-100 rounded-lg ${disabled ? 'opacity-50 pointer-events-none' : ''} ${className ?? ''}`}>
      {TOOLS.map((tool) => {
        const Icon = tool.icon
        const isActive = activeTool === tool.key
        const count = tool.countKey ? counts[tool.countKey] ?? 0 : 0

        return (
          <button key={tool.key} onClick={() => onToolChange(tool.key)} disabled={disabled}
            title={`${tool.label} (${tool.shortcut})`}
            className={`relative flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
              isActive ? 'bg-white text-slate-900 shadow-sm' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'
            }`}>
            <Icon className="w-4 h-4" />
            <span className="hidden sm:inline">{tool.label}</span>
            {count > 0 && (
              <span className={`absolute -top-1 -right-1 w-4 h-4 rounded-full text-[10px] font-bold flex items-center justify-center ${
                isActive ? 'bg-indigo-500 text-white' : 'bg-gray-400 text-white'
              }`}>{count}</span>
            )}
          </button>
        )
      })}

      {/* Precision crosshair toggle */}
      <div className="w-px h-6 bg-gray-300 mx-0.5" />
      <button onClick={() => onPrecisionModeChange?.(!precisionMode)}
        title={`Precision mode ${precisionMode ? 'ON' : 'OFF'} (P)`}
        className={`flex items-center gap-1 px-2 py-1.5 rounded-md text-xs font-medium transition-colors ${
          precisionMode ? 'bg-violet-100 text-violet-700' : 'text-gray-500 hover:text-gray-800 hover:bg-gray-200'
        }`}>
        <Crosshair className="w-3.5 h-3.5" />
        <span className="hidden sm:inline">Precision</span>
      </button>

      {/* Reset */}
      <div className="w-px h-6 bg-gray-300 mx-1" />
      <button onClick={onReset} disabled={disabled || total === 0}
        className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
          total > 0 ? 'text-gray-600 hover:text-gray-900 hover:bg-gray-200' : 'text-gray-400 cursor-not-allowed'
        }`}>
        <RotateCcw className="w-4 h-4" />
        <span className="hidden sm:inline">Reset</span>
      </button>
    </div>
  )
}
