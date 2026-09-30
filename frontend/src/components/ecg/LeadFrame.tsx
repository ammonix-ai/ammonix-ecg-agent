interface LeadFrameProps {
  leadName: string
  isActiveRhythm?: boolean
  onClick?: () => void
  disabled?: boolean
  children: React.ReactNode
  className?: string
}

/**
 * LeadFrame — Transparent window frame for a single lead
 *
 * Provides:
 * - Transparent background (grid shows through from ECGGridPaper)
 * - Lead label in top-left corner
 * - Hover state for expandable leads
 *
 * The frame has NO grid — it's transparent so the unified
 * grid from ECGGridPaper shows through.
 */
export function LeadFrame({
  leadName,
  isActiveRhythm = false,
  onClick,
  disabled = false,
  children,
  className,
}: LeadFrameProps) {
  const isClickable = onClick && !disabled

  return (
    <div
      className={`
        relative bg-transparent border overflow-hidden
        ${isClickable ? 'cursor-pointer hover:border-indigo-500 hover:bg-white/30 transition-all' : ''}
        ${disabled ? 'cursor-not-allowed' : ''}
        ${className ?? ''}
      `}
      style={{ borderColor: 'transparent' }}
      onClick={isClickable ? onClick : undefined}
    >
      {/* Lead label */}
      <div className="absolute top-0.5 left-1 z-10">
        <span
          className={`
            px-1 py-0.5 rounded text-[10px] font-bold shadow-sm
            ${isActiveRhythm
              ? 'bg-indigo-700 text-white'
              : 'bg-white/90 text-slate-700'}
          `}
        >
          {leadName}
        </span>
      </div>

      {/* Trace content */}
      <div className="absolute inset-0">
        {children}
      </div>

      {/* Expand hint on hover */}
      {isClickable && (
        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-white/10">
          <span className="text-xs font-medium text-indigo-700 bg-white/90 px-2 py-1 rounded shadow-sm">
            Click to expand
          </span>
        </div>
      )}
    </div>
  )
}
