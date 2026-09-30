interface CrosshairOverlayProps {
  mouseX: number
  mouseY: number
  containerWidth: number
  containerHeight: number
  visibleStart: number
  pixelsPerSecond: number
  pixelsPerMillivolt: number
  className?: string
}

/**
 * CrosshairOverlay — Full edge-to-edge tracking crosshair
 *
 * Vertical line shows time position, horizontal shows amplitude.
 * Colored pill labels at bottom (time) and left (mV).
 * pointer-events-none so it doesn't block tool interactions.
 */
export function CrosshairOverlay({
  mouseX, mouseY,
  containerWidth, containerHeight,
  visibleStart, pixelsPerSecond, pixelsPerMillivolt,
  className,
}: CrosshairOverlayProps) {
  const x = Math.max(0, Math.min(mouseX, containerWidth))
  const y = Math.max(0, Math.min(mouseY, containerHeight))

  const time = visibleStart + x / pixelsPerSecond
  const mV = (containerHeight / 2 - y) / pixelsPerMillivolt

  const timeLabel = time < 1 ? `${(time * 1000).toFixed(0)}ms` : time < 10 ? `${time.toFixed(2)}s` : `${time.toFixed(1)}s`
  const mvLabel = `${mV >= 0 ? '+' : ''}${mV.toFixed(2)}`

  return (
    <div className={`absolute inset-0 pointer-events-none z-10 ${className ?? ''}`}>
      {/* Vertical line */}
      <div className="absolute top-0 bottom-0 w-px bg-indigo-500/40" style={{ left: x }} />
      {/* Horizontal line */}
      <div className="absolute left-0 right-0 h-px bg-indigo-500/40" style={{ top: y }} />

      {/* Time label (bottom) */}
      <div className="absolute flex flex-col items-center" style={{ left: x, bottom: 2, transform: 'translateX(-50%)' }}>
        <span className="px-1.5 py-0.5 text-[10px] font-medium text-white bg-indigo-500 rounded-sm whitespace-nowrap shadow-sm">
          {timeLabel}
        </span>
      </div>

      {/* mV label (left) */}
      <div className="absolute flex items-center" style={{ top: y, left: 2, transform: 'translateY(-50%)' }}>
        <span className="px-1.5 py-0.5 text-[10px] font-medium text-white bg-indigo-500 rounded-sm whitespace-nowrap shadow-sm">
          {mvLabel}
        </span>
      </div>
    </div>
  )
}
