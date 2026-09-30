import { apiGet, apiPost } from '@/api/client'

/**
 * Diagnostics Service — V5 ML pipeline data endpoints
 *
 * Wraps /api/diagnostics/* for evaluation, calibration, templates, RLVR zones.
 */

/**
 * Status sentinel emitted by `/api/diagnostics/*` routes when the underlying
 * artifact is missing.
 *
 * Backend C0.2 Phase 2 normalized 3 historical strings into a single
 * `unavailable` canonical form, but kept the legacy literals as Pydantic
 * Literal aliases for backward-compat. Frontend should treat any of the 4
 * as "no data to render" via `isUnavailable(data)`.
 */
export type DiagnosticsUnavailableStatus =
  | 'unavailable'
  | 'not_computed'
  | 'not_available'
  | 'no_data'

/** True when a diagnostics route returned a sentinel rather than data. */
export function isUnavailable(data: unknown): boolean {
  if (data == null) return true
  if (typeof data !== 'object') return false
  const status = (data as { status?: unknown }).status
  if (typeof status !== 'string') return false
  return (
    status === 'unavailable' ||
    status === 'not_computed' ||
    status === 'not_available' ||
    status === 'no_data'
  )
}

// ── 4-Column Evaluation ──

export interface EvalDiagnosisRow {
  diagnosis: string
  /** Number of positive examples */
  n?: number
  threshold?: number
  auroc?: number
  f1_raw: number
  f1_det: number
  f1_m1def: number
  f1_m1full: number
  prec_raw: number
  rec_raw: number
  prec_det: number
  rec_det: number
  prec_m1def?: number
  rec_m1def?: number
  prec_m1full?: number
  rec_m1full?: number
}

export interface EvaluationResult {
  status: string
  split_name?: string
  n_patients?: number
  macro_f1_raw?: number
  macro_f1_det?: number
  macro_f1_m1def?: number
  macro_f1_m1full?: number
  per_diagnosis?: EvalDiagnosisRow[]
  columns?: string[]
}

export async function getEvaluation(): Promise<EvaluationResult> {
  return apiGet<EvaluationResult>('/api/diagnostics/evaluation')
}

// ── Calibration ──

export interface CalibrationSummary {
  status: string
  per_class?: Record<string, {
    method: string
    ece_before: number
    ece_after: number
  }>
}

export async function getCalibrationSummary(): Promise<CalibrationSummary> {
  return apiGet<CalibrationSummary>('/api/diagnostics/calibration')
}

export async function getCalibrationCurve(diagnosis: string): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>(`/api/diagnostics/calibration/${encodeURIComponent(diagnosis)}`)
}

// ── RLVR Zones ──

export interface RLVRZoneSummary {
  status: string
  zones?: Record<string, {
    f1_threshold: number
    zone_lower: number
    n_fn: number
    n_tn: number
  }>
}

export async function getRLVRZones(): Promise<RLVRZoneSummary> {
  return apiGet<RLVRZoneSummary>('/api/diagnostics/rlvr-zones')
}

export async function getRLVRZoneDetail(diagnosis: string, maxTn = 50): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>(`/api/diagnostics/rlvr-zones/${encodeURIComponent(diagnosis)}?max_tn=${maxTn}`)
}

// ── Waveform Templates ──

export interface TemplateSummary {
  status: string
  n_templates?: number
  diagnoses?: string[]
  vector_length?: number
}

export async function getTemplateSummary(): Promise<TemplateSummary> {
  return apiGet<TemplateSummary>('/api/diagnostics/templates')
}

export async function getTemplateForDiagnosis(diagnosis: string): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>(`/api/diagnostics/templates/${encodeURIComponent(diagnosis)}`)
}

export interface SaveTemplateRequest {
  leads: Record<string, {
    components: Array<{
      wave: string
      amplitude: number
      center: number
      width: number
    }>
  }>
  notes?: string
  source?: string
}

export interface SaveTemplateResponse {
  status: string
  diagnosis: string
  source: string
  n_leads: number
}

export async function saveTemplate(
  diagnosis: string,
  data: SaveTemplateRequest,
): Promise<SaveTemplateResponse> {
  return apiPost<SaveTemplateResponse>(
    `/api/diagnostics/templates/${encodeURIComponent(diagnosis)}`,
    data,
  )
}

// ── Patient Template Deviations ──

export interface TemplateDeviation {
  patient_id: string
  deviations: Record<string, unknown>
}

export async function getPatientTemplateDeviations(patientId: string, topN = 3): Promise<TemplateDeviation> {
  return apiGet<TemplateDeviation>(`/api/diagnostics/template-deviations/${encodeURIComponent(patientId)}?top_n=${topN}`)
}

// ── Template Outliers ──
//
// Backend route lacks `response_model=`; mirrored from
// `backend/routers/diagnostics.py:539-625` (returns dict[str, Any]).
// Response shape:
//   { status: "computed", outliers: [...], total_scanned, total_outliers, min_zscore_used }
//   { status: "no_templates", outliers: [], total_scanned: 0 }

export interface TemplateOutlierFeature {
  key: string
  zscore: number
}

export interface TemplateOutlierRow {
  patientId: string
  displayId: string
  diagnosis: string
  maxZscore: number
  distance: number
  templatePatients: number
  topDeviations: TemplateOutlierFeature[]
  cohort: string
}

export interface TemplateOutliersResponse {
  status: string
  outliers?: TemplateOutlierRow[]
  total_scanned?: number
  total_outliers?: number
  min_zscore_used?: number
}

export async function getTemplateOutliers(
  limit = 50,
  minZscore = 2.0,
): Promise<TemplateOutliersResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    min_zscore: String(minZscore),
  })
  return apiGet<TemplateOutliersResponse>(
    `/api/diagnostics/template-outliers?${params.toString()}`,
  )
}

// ── Diagnostic Tools ──

export async function getConfusionPairs(topN = 10): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>(`/api/diagnostics/confusion-pairs?top_n=${topN}`)
}

export async function getDiagnosisRankings(): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>('/api/diagnostics/diagnosis-rankings')
}

// --- Cost Curves ---

export interface CostCurvePoint {
  threshold: number
  cost: number
  fp_count: number
  fn_count: number
}

export interface CostCurveResult {
  diagnosis: string
  fn_cost: number
  fp_cost: number
  points: CostCurvePoint[]
  optimal_threshold: number
  optimal_cost: number
  n_positive: number
  n_total: number
}

export async function getCostCurve(
  diagnosis: string,
  fnCost?: number,
  fpCost?: number,
): Promise<CostCurveResult> {
  const params = new URLSearchParams()
  if (fnCost !== undefined) params.set('fn_cost', String(fnCost))
  if (fpCost !== undefined) params.set('fp_cost', String(fpCost))
  const qs = params.toString()
  return apiGet<CostCurveResult>(
    `/api/diagnostics/cost-curves/${encodeURIComponent(diagnosis)}${qs ? '?' + qs : ''}`,
  )
}

// --- Threshold Optimization ---

export interface OptimizeThresholdsRequest {
  n_trials?: number
  use_calibrated?: boolean
  save_to_rules?: boolean
  timeout?: number
}

export interface OptimizeThresholdsResult {
  thresholds: Record<string, number>
  sample_f1: number
  baseline_sample_f1: number
  improvement: number
  n_trials: number
  duration_ms: number
  saved_to_rules: boolean
}

export async function optimizeThresholds(
  req: OptimizeThresholdsRequest,
): Promise<OptimizeThresholdsResult> {
  return apiPost<OptimizeThresholdsResult>('/api/diagnostics/thresholds/optimize', req)
}

// ── CV Results ──
//
// /api/diagnostics/cv-results returns one of:
//   { status: "available", source: "cv_artifacts" | "experiment", n_patients, n_work,
//     n_holdout, n_folds, metrics, per_fold, thresholds, ... }
//   { status: "not_computed", hint: "..." }

export interface CVResultsResponse {
  status: string
  source?: 'cv_artifacts' | 'experiment'
  experiment_id?: string
  experiment_name?: string
  n_patients?: number
  n_work?: number
  n_holdout?: number
  n_folds?: number
  metrics?: Record<string, unknown> | null
  per_fold?: unknown[] | null
  thresholds?: Record<string, number> | null
  hint?: string
}

export async function getCVResults(): Promise<CVResultsResponse> {
  return apiGet<CVResultsResponse>('/api/diagnostics/cv-results')
}

// ── Concordance ──
//
// /api/diagnostics/concordance returns one of:
//   { status: "computed", has_zscores: true,  concordance: {...}, safe_zones: [...], metadata: {...} }
//   { status: "partial",  has_zscores: false, concordance: {...}, safe_zones: [],     metadata: {...} }
//   { status: "not_available", has_zscores: false, concordance: {}, safe_zones: [],   metadata: {...}, hint: "..." }

export interface ConcordanceRow {
  ppv: number
  npv: number
  concordance_score: number
  p_value: number
  n_positive: number
}

export interface SafeZoneRow {
  diagnosis: string
  p_threshold: number
  z_threshold: number
  npv: number
  n_negatives: number
  n_zone: number
  rule_text: string
}

export interface ConcordanceMetadata {
  n_patients: number
  n_classes: number
  cohort: string
  note?: string
}

export interface ConcordanceResponse {
  status: string
  has_zscores: boolean
  concordance: Record<string, ConcordanceRow>
  safe_zones: SafeZoneRow[]
  metadata: ConcordanceMetadata
  hint?: string
}

export async function getConcordance(cohort = 'Ar'): Promise<ConcordanceResponse> {
  return apiGet<ConcordanceResponse>(
    `/api/diagnostics/concordance?cohort=${encodeURIComponent(cohort)}`,
  )
}

// ── Conformal Prediction Sets ──
//
// /api/diagnostics/conformal-sets returns one of:
//   { status: "computed", alpha, target_coverage, empirical_coverage, mean_set_size,
//     per_diagnosis: [...], duration_ms }
//   { status: "not_computed", hint: "..." }

export interface ConformalDiagnosisRow {
  diagnosis: string
  coverage: number
  avg_set_size: number
  n_positive: number
}

export interface ConformalSetsResponse {
  status: string
  alpha?: number
  target_coverage?: number
  empirical_coverage?: number
  mean_set_size?: number
  per_diagnosis?: ConformalDiagnosisRow[]
  duration_ms?: number
  hint?: string
}

export async function getConformalSets(alpha = 0.05): Promise<ConformalSetsResponse> {
  return apiGet<ConformalSetsResponse>(`/api/diagnostics/conformal-sets?alpha=${alpha}`)
}
