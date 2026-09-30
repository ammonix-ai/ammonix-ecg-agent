import { ECG_GRID_COLORS } from '@/types/ecg'
import type { PaperSpeed, GainSetting } from '@/types/ecg'
import { getGridSpacing } from '@/types/ecg'

interface CalibrationPulseProps {
  /** Height in pixels (1mV at current gain) */
  heightPx: number
  /** Width in pixels (200ms at current paper speed) */
  widthPx: number
  showLabel?: boolean
  className?: string
}

/**
 * CalibrationPulse — 1mV reference marker
 *
 * Standard: 1mV height x 200ms width
 * At 25mm/s, 10mm/mV: 10mm x 5mm (one large box each)
 * Shape: └──┐ (baseline up, across, down to baseline)
 */
export function CalibrationPulse({
  heightPx,
  widthPx,
  showLabel = true,
  className,
}: CalibrationPulseProps) {
  const strokeWidth = ECG_GRID_COLORS.traceWidth
  const padding = strokeWidth + 2

  const totalWidth = widthPx + padding * 2
  const totalHeight = heightPx + padding * 2

  return (
    <div className={`flex flex-col items-center ${className ?? ''}`}>
      <svg
        width={totalWidth}
        height={totalHeight}
        className="block"
      >
        <path
          d={`
            M ${padding} ${totalHeight - padding}
            L ${padding} ${padding}
            L ${widthPx + padding} ${padding}
            L ${widthPx + padding} ${totalHeight - padding}
          `}
          fill="none"
          stroke={ECG_GRID_COLORS.trace}
          strokeWidth={strokeWidth}
          strokeLinecap="square"
          strokeLinejoin="miter"
        />
      </svg>

      {showLabel && (
        <span className="text-[9px] font-medium text-gray-600 mt-0.5">
          1mV
        </span>
      )}
    </div>
  )
}

/**
 * Calculate calibration pulse pixel dimensions for given settings.
 *
 * At standard settings (25mm/s, 10mm/mV):
 * - Height: 10mm = 1mV x 10mm/mV
 * - Width: 5mm = 200ms x 25mm/s / 1000
 */
export function getCalibrationDimensions(
  paperSpeed: PaperSpeed = 25,
  gain: GainSetting = 10,
): { height: number; width: number } {
  const spacing = getGridSpacing(paperSpeed, gain)
  return {
    height: spacing.pixelsPerMv,
    width: (200 / 1000) * spacing.pixelsPerSecond,
  }
}
