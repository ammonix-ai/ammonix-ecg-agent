import { ECG_GRID_COLORS, LARGE_BOX_PX } from '@/types/ecg'
import type { GridMode } from '@/types/ecg'

interface ECGGridPaperProps {
  children: React.ReactNode
  gridMode?: GridMode
  /** Custom major-gridline size in px (defaults to LARGE_BOX_PX ~18.9px) */
  boxSize?: number
  /** Horizontal zoom factor — stretches X grid spacing */
  zoomX?: number
  /** X offset (px) for panning alignment */
  gridOffsetX?: number
  /** Y offset (px) to align a major gridline with 0mV baseline */
  gridOffsetY?: number
  style?: React.CSSProperties
  className?: string
}

/**
 * ECGGridPaper — IEC 60601-2-25 compliant pink grid background
 *
 * Foundation layer for ECG visualization.
 * Lead frames sit on top as transparent windows.
 *
 * Grid specs:
 * - Minor lines: every 1mm (light pink, 0.5px)
 * - Major lines: every 5mm (darker pink, 1.0px)
 * - Background: warm off-white (#FFF5F5)
 */
export function ECGGridPaper({
  children,
  gridMode = 'full',
  boxSize,
  zoomX = 1,
  gridOffsetX = 0,
  gridOffsetY = 0,
  style,
  className,
}: ECGGridPaperProps) {
  const by = boxSize ?? LARGE_BOX_PX
  const bx = (boxSize ?? LARGE_BOX_PX) * zoomX
  const sy = (boxSize ?? LARGE_BOX_PX) / 5
  const sx = ((boxSize ?? LARGE_BOX_PX) / 5) * zoomX

  const oX = gridOffsetX ? `${gridOffsetX}px` : '0'
  const oY = gridOffsetY ? `${gridOffsetY}px` : '0'

  let gridStyle: React.CSSProperties

  if (gridMode === 'full') {
    gridStyle = {
      backgroundColor: ECG_GRID_COLORS.background,
      backgroundImage: `
        linear-gradient(to right, ${ECG_GRID_COLORS.major} 1px, transparent 1px),
        linear-gradient(to bottom, ${ECG_GRID_COLORS.major} 1px, transparent 1px),
        linear-gradient(to right, ${ECG_GRID_COLORS.minor} 0.5px, transparent 0.5px),
        linear-gradient(to bottom, ${ECG_GRID_COLORS.minor} 0.5px, transparent 0.5px)
      `,
      backgroundSize: `${bx}px ${by}px, ${bx}px ${by}px, ${sx}px ${sy}px, ${sx}px ${sy}px`,
      backgroundPosition: `${oX} ${oY}, ${oX} ${oY}, ${oX} ${oY}, ${oX} ${oY}`,
      ...style,
    }
  } else if (gridMode === 'major') {
    gridStyle = {
      backgroundColor: ECG_GRID_COLORS.background,
      backgroundImage: `
        linear-gradient(to right, ${ECG_GRID_COLORS.major} 1px, transparent 1px),
        linear-gradient(to bottom, ${ECG_GRID_COLORS.major} 1px, transparent 1px)
      `,
      backgroundSize: `${bx}px ${by}px, ${bx}px ${by}px`,
      backgroundPosition: `${oX} ${oY}, ${oX} ${oY}`,
      ...style,
    }
  } else {
    gridStyle = {
      backgroundColor: '#FFFFFF',
      ...style,
    }
  }

  return (
    <div
      className={`relative overflow-hidden ${className ?? ''}`}
      style={gridStyle}
    >
      {children}
    </div>
  )
}
