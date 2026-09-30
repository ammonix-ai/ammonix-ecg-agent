/**
 * Projection types for 3D Universe visualization.
 *
 * Ported from FEBE (frontend/src/types/projection.ts).
 *
 * Open build: the 'clinical' projection method and the 'fold'/'fold_simple'
 * color modes were dropped. 'clinical' was an alias for PCA on this data
 * (identical sklearn call server-side) and the CV-only variant needed a
 * private embedding module. FoldMode itself stays — it is still the wire type
 * for the backend's `fold_mode` query param.
 *
 * There is no training/test distinction on a point. The published universe is
 * scored with the mean of all five folds for every row and ships no fold
 * membership, so nothing here can say which records were held out. `cohort`
 * (the four source cohorts) is what actually distinguishes them.
 *
 * All interfaces are defined manually (no OpenAPI codegen in UI_V2).
 */

export type ProjectionMethod = 'tsne' | 'umap' | 'pca'

export type FoldMode = 'all' | 'fold_0' | 'fold_1' | 'fold_2' | 'fold_3' | 'fold_4' | 'holdout'

/**
 * 'accuracy' and 'accuracy_pipeline' were removed: they split every point into
 * exact-match vs differs, which is the same partition Error Highlight makes —
 * except Error Highlight also says HOW it differs (over-called, missed, or
 * both). Two controls answering the same question, one of them less
 * informative, is a choice the reader should not have to make.
 */
export type ColorByMode =
  | 'cohort'
  | 'diagnosis'
  | 'diagnosis_predicted'
  | 'error_highlight'
  | 'family'

export interface UniverseCluster {
  id: string
  name: string
  color: string
  count: number
}

export interface ColorPalette {
  xgbCorrect: string
  xgbIncorrect: string
  xgbUnknown: string
}

export interface DisplaySettings {
  pointSize: number
  selectedPointSize: number
  neighborPointSize: number
  opacity: number
  showEdgeOutlines: boolean
  showGrid: boolean
  showDiagnosisLabels: boolean
  showPinnedLabels: boolean
  showCentroids: boolean
  showTribeBoundaries: boolean
  centroidSize: number
  centroidOpacity: number
  boundaryOpacity: number
}

export interface ViewerSettings {
  colors: ColorPalette
  display: DisplaySettings
  diagnosisColorOverrides: Record<string, string>
}

export const DEFAULT_COLOR_PALETTE: ColorPalette = {
  xgbCorrect: '#22C55E',
  xgbIncorrect: '#EF4444',
  xgbUnknown: '#64748B',
}

export const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  pointSize: 0.05,
  selectedPointSize: 0.08,
  neighborPointSize: 0.06,
  opacity: 1.0,
  showEdgeOutlines: true,
  showGrid: true,
  showDiagnosisLabels: true,
  showPinnedLabels: true,
  showCentroids: false,
  showTribeBoundaries: false,
  centroidSize: 0.3,
  centroidOpacity: 0.7,
  boundaryOpacity: 0.15,
}

export const DEFAULT_VIEWER_SETTINGS: ViewerSettings = {
  colors: DEFAULT_COLOR_PALETTE,
  display: DEFAULT_DISPLAY_SETTINGS,
  diagnosisColorOverrides: {},
}

/**
 * Colours for the `cohort` grading mode, assigned by sorted cohort name.
 * Four cohorts ship today (Apr28 multi-source, MIMIC SR normals, LVEF EF<40
 * cases, LVEF EF>=55 controls); the palette cycles if that ever grows.
 */
export const COHORT_COLOR_PALETTE: string[] = [
  '#38BDF8', // sky
  '#F59E0B', // amber
  '#A855F7', // purple
  '#10B981', // emerald
  '#EF4444', // red
  '#EC4899', // pink
]

export const DIAGNOSIS_COLOR_PALETTE: string[] = [
  '#E53935', '#D81B60', '#8E24AA', '#5E35B1',
  '#3949AB', '#1E88E5', '#039BE5', '#00ACC1',
  '#00897B', '#43A047', '#7CB342', '#C0CA33',
  '#FDD835', '#FFB300', '#FB8C00', '#F4511E',
  '#6D4C41', '#546E7A', '#AB47BC', '#5C6BC0',
  '#26A69A', '#9CCC65', '#FF7043', '#78909C',
  '#EC407A', '#42A5F5', '#66BB6A', '#FFCA28',
]

// =============================================================================
// Projection data interfaces (manual — no OpenAPI codegen)
// =============================================================================

export interface ProjectionPoint {
  /**
   * The record's PhysioNet recording id when it has one (HR07445), else its
   * display id. Same record as `displayId` — see `pointIdLabel` in
   * components/universe/recordingLink.ts, which shows both when they differ.
   */
  patientId: string
  x: number
  y: number
  z: number
  primaryDiagnosis: string
  diagnoses: string[]
  cohort: string
  color: string
  symbol: string
}

export interface EnrichedProjectionPoint extends ProjectionPoint {
  /** Stable public key, SS-NNNNN. Present on every published row. */
  displayId?: string
  xgbCorrect?: boolean
  xgbPredictions: string[]
  pipelineCorrect?: boolean
  pipelinePredictions: string[]
  tribes: string[]
  multiTribe: boolean
  /** Demographics are not published with the universe: the wire value is null. */
  age?: number | null
  sex?: string | null
  modelClassClean: boolean
  // Cohort overlay additions
  isSupplement?: boolean
  sourceCohort?: string
  // Source-safe universe additions
  source?: string
  subject?: string
  topPrediction?: string
  maxScore?: number
}

export interface SourceSafeProjectionMetadata {
  modelFamily?: string
  modelVersion?: string
  modelHeadline?: string
  macroPrimaryAuc?: number
  macroFinalAuc?: number
  dataDir?: string
  packageDir?: string
  featureCount?: number
  nClasses?: number
  thresholds?: Record<string, number>
  diagnoses?: Array<Record<string, any>>
  sourceCounts?: Record<string, number>
  cohortCounts?: Record<string, number>
}

export interface EnrichedProjectionResponse {
  method: string
  points: EnrichedProjectionPoint[]
  totalPatients: number
  cohortsIncluded: string[]
  hyperparameters: Record<string, any>
  diagnosisColors: Record<string, string>
  cached: boolean
  /** PCA / clinical projections include human-readable axis labels. */
  axisLabels?: string[]
  sourceSafe?: SourceSafeProjectionMetadata
}

export interface CVEnrichedProjectionResponse extends EnrichedProjectionResponse {
  foldMode: string
  axisLabels: string[]
}

export interface AvailableProjection {
  method: string
  cached: boolean
  totalPatients: number
  timestamp?: string
}

export interface CohortLatticeStatus {
  cohort_id: number
  cohort_name: string
  record_count: number
  has_qpsi: boolean
  qpsi_file: string | null
  qpsi_records: number
  has_training: boolean
  training: {
    cohort_id: number
    cohort_name: string
    n_patients: number
    n_classes: number
    n_folds: number
    holdout_size: number
    status: string
    trained_at: string | null
    macro_f1: number | null
    macro_auroc: number | null
    classes: string[] | null
    model_family?: string
    version?: string
  } | null
  source_safe?: boolean
  cache_built?: boolean
}
