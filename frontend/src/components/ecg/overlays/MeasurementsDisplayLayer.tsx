import { CaliperDisplay } from './CaliperOverlay'
import { RulerDisplay } from './RulerOverlay'
import { AnnotationMarker } from './DrawOverlay'
import type { CaliperMeasurement, RulerMeasurement, Annotation } from '@/types/ecgTools'

interface MeasurementsDisplayLayerProps {
  caliperMeasurements: CaliperMeasurement[]
  rulerMeasurements: RulerMeasurement[]
  annotations: Annotation[]
  onRemoveCaliper: (id: string) => void
  onRemoveRuler: (id: string) => void
  onRemoveAnnotation: (id: string) => void
}

/**
 * MeasurementsDisplayLayer — Renders all completed measurements
 *
 * Sits at z-20 with pointer-events-none on root, but pointer-events-auto
 * on individual labels/buttons for delete.
 *
 * HR Calculator manages its own rendering and wipes on deactivate.
 */
export function MeasurementsDisplayLayer({
  caliperMeasurements,
  rulerMeasurements,
  annotations,
  onRemoveCaliper,
  onRemoveRuler,
  onRemoveAnnotation,
}: MeasurementsDisplayLayerProps) {
  return (
    <div className="absolute inset-0 pointer-events-none z-20">
      {caliperMeasurements.map((m) => (
        <CaliperDisplay key={m.id} measurement={m} onRemove={() => onRemoveCaliper(m.id)} />
      ))}
      {rulerMeasurements.map((m) => (
        <RulerDisplay key={m.id} measurement={m} onRemove={() => onRemoveRuler(m.id)} />
      ))}
      {annotations.map((a) => (
        <AnnotationMarker key={a.id} annotation={a} onRemove={() => onRemoveAnnotation(a.id)} />
      ))}
    </div>
  )
}
