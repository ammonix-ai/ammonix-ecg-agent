/**
 * ECG Tool Types — measurement, annotation, and overlay types
 * Ported from FEBE frontend/src/types/ecgTools.ts
 */

export type CaliperAxis = 'x' | 'y'
export type CaliperSnapMode = 'free' | 'grid'

export interface SnapPoint {
  x: number
  y: number
  source: 'caliper' | 'ruler' | 'annotation' | 'hr-calc'
  sourceId: string
}

export function findNearestSnapPoint(
  x: number, y: number, snapPoints: SnapPoint[], snapRadius = 10
): SnapPoint | null {
  let nearest: SnapPoint | null = null
  let minDist = snapRadius
  for (const sp of snapPoints) {
    const dist = Math.sqrt((sp.x - x) ** 2 + (sp.y - y) ** 2)
    if (dist < minDist) { minDist = dist; nearest = sp }
  }
  return nearest
}

export interface CaliperMeasurement {
  id: string
  axis: CaliperAxis
  startX: number
  endX: number
  y: number
  milliseconds: number
  startY?: number
  endY?: number
  millivolts?: number
}

export interface RulerMeasurement {
  id: string
  startX: number
  startY: number
  endX: number
  endY: number
  angleDegrees: number
  distancePx: number
}

export interface Annotation {
  id: string
  x: number
  y: number
  type: 'marker' | 'label' | 'stroke'
  label: string
  color: string
  strokePoints?: { x: number; y: number }[]
}

export function calculateAngle(x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1
  const dy = y1 - y2 // screen Y inverted
  return Math.round(Math.atan2(dy, dx) * (180 / Math.PI) * 10) / 10
}

export function calculateDistance(x1: number, y1: number, x2: number, y2: number): number {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
}

export const ANNOTATION_PRESETS = [
  { label: 'P wave', color: '#3B82F6' },
  { label: 'QRS', color: '#8B5CF6' },
  { label: 'T wave', color: '#10B981' },
  { label: 'U wave', color: '#06B6D4' },
  { label: 'ST elevation', color: '#EF4444' },
  { label: 'ST depression', color: '#F59E0B' },
  { label: 'Artifact', color: '#6B7280' },
  { label: 'Ectopic', color: '#EC4899' },
  { label: 'Pause', color: '#F97316' },
] as const
