import { apiGet, apiPost } from '@/api/client';
import type {
  CohortLatticeStatus,
  EnrichedProjectionResponse,
} from '@/types/projection';

/**
 * CV Pipeline Service — 5-fold cross-validation lifecycle and SSE streaming.
 *
 * Endpoints (all under /api/admin/cv/*):
 * - GET    /files                       — list available QPSI JSONL files
 * - GET    /status                      — current CV worker status
 * - POST   /stop                        — cancel running CV
 * - GET    /results                     — last completed CV results
 * - POST   /rlvr/start                  — start RLVR error correction
 * - GET    /rlvr/status                 — current RLVR worker status
 * - GET    /classifier-metrics          — CV-derived ClassifierPage metrics
 * - GET    /projection/{method}         — 3D projection from OOF probs (per-cohort)
 * - GET    /lattice-status              — lattice status across all cohorts
 *
 * Open build: `/start`, `/stream` (SSE) and `/projection/{method}/overlay` are
 * not served by the open backend, so their client wrappers were removed.
 *
 * V3.6b: removed `getCVResults` (`/api/admin/cv/results`) — orphan with
 * zero V3 consumers and a name-collision against
 * `diagnosticsService.getCVResults` (`/api/diagnostics/cv-results`, the route
 * ClassifierPage actually wires). Two different backend routes; only the
 * diagnostics one is live in V3.
 *
 * V3.6d: typed end-to-end. `CVStartRequest` and `OverlayProjectionRequest`
 * inputs are schema-derived (the only routes with `response_model=` style
 * Pydantic input wiring). Response types are service-local interfaces
 * mirroring backend dict shapes — admin/cv.py omits `response_model=` on
 * read routes so the regenerated schema.d.ts ships `unknown` for those.
 * Flagged as Tier 4 backend hygiene candidate.
 */

// =============================================================================
// Service-local response types (admin/cv.py omits response_model=)
// =============================================================================

export interface QPSIFile {
  filename: string;
  records: number;
  sizeMb: number;
  modified: string;
  isCohort: boolean;
}

export interface QPSIFilesResponse {
  files: QPSIFile[];
}

/**
 * Status payload from `worker_get_status().to_dict()`. Idle path returns
 * `{status: "idle"}` only; running/complete paths return the full set.
 */
export interface CVStatus {
  status: string;
  current_fold?: number;
  total_folds?: number;
  current_step?: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  n_patients?: number;
  n_holdout?: number;
  n_working?: number;
  n_classes?: number;
  macro_f1?: number | null;
  macro_auroc?: number | null;
  cohort_id?: number | null;
  cohort_name?: string | null;
}

export interface CVRLVRStatus {
  status: string;
  [key: string]: unknown;
}

/** Per-diagnosis metrics row from `_compute_live_metrics` / file-based response. */
export interface CVPerDiagnosis {
  auroc: number;
  f1: number;
  precision: number;
  recall: number;
  support: number;
  is_noisy: boolean;
  f1_ci?: [number, number];
}

export interface CVThresholdRow {
  diagnosis: string;
  threshold: number;
  rlvr_threshold?: number | null;
  category: string;
}

export interface CVCohortStats {
  total: { count: number };
  working: { count: number };
  holdout: { count: number };
}

export interface CVConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface CVClassifierMetrics {
  source: string;
  sourceFiles: string[];
  auroc: Record<string, number>;
  f1: Record<string, number>;
  macroF1: number;
  macroAuroc: number;
  diagnosisLabels: string[];
  perDiagnosis: Record<string, CVPerDiagnosis>;
  thresholds: CVThresholdRow[];
  cohortStats: CVCohortStats;
  nFolds: number;
  rlvrRules: unknown[];
  confusionMatrix: CVConfusionMatrix | null;
}

export interface LatticeStatusResponse {
  cohorts: CohortLatticeStatus[];
}

/**
 * SSE event payload from `createCVEventSource`. Backend dispatches discrete
 * `cv_status`, `cv_progress`, `fold_start`, `fold_complete`,
 * `threshold_optimized`, `holdout_evaluated`, `cv_complete`, `cv_error`,
 * plus the synthetic `keepalive` and `raw` types we synthesize client-side.
 */
export type CVEventType =
  | 'cv_status'
  | 'cv_progress'
  | 'fold_start'
  | 'fold_complete'
  | 'threshold_optimized'
  | 'holdout_evaluated'
  | 'cv_complete'
  | 'cv_error'
  | 'keepalive'
  | 'raw'
  | 'message';

/**
 * The `data` payload is loosely-typed because backend SSE events ship
 * heterogeneous dicts (cv_progress: {detail, step}; fold_complete:
 * {fold, total}; cv_error: {error}; etc.) and the synthetic 'raw'
 * fallback ships a string. Consumers field-alias defensively.
 */
export interface CVEvent {
  type: CVEventType | string;
  data: any;
}

// =============================================================================
// Endpoints
// =============================================================================

export async function listQPSIFiles(): Promise<QPSIFilesResponse> {
  return apiGet('/api/admin/cv/files');
}

// Open build: startCVPipeline (POST /api/admin/cv/start) and
// createCVEventSource (GET /api/admin/cv/stream, SSE) were removed. Training is
// not driven from the open UI — the universe ships pre-trained, and neither
// endpoint exists on the open backend.

export async function getCVStatus(): Promise<CVStatus> {
  return apiGet('/api/admin/cv/status');
}

export async function stopCV(): Promise<CVStatus> {
  return apiPost('/api/admin/cv/stop');
}

export async function startCVRLVR(cohortId?: number): Promise<CVRLVRStatus> {
  const qs = cohortId != null ? `?cohort_id=${cohortId}` : '';
  return apiPost(`/api/admin/cv/rlvr/start${qs}`);
}

export async function getCVRLVRStatus(): Promise<CVRLVRStatus> {
  return apiGet('/api/admin/cv/rlvr/status');
}

export async function getCVClassifierMetrics(): Promise<CVClassifierMetrics> {
  return apiGet('/api/admin/cv/classifier-metrics');
}

export async function getCVProjection(
  method: string = 'tsne',
  params?: { perplexity?: number; n_neighbors?: number },
): Promise<EnrichedProjectionResponse> {
  const qs = new URLSearchParams();
  if (params?.perplexity) qs.set('perplexity', String(params.perplexity));
  if (params?.n_neighbors) qs.set('n_neighbors', String(params.n_neighbors));
  const query = qs.toString();
  return apiGet(`/api/admin/cv/projection/${method}${query ? `?${query}` : ''}`);
}

// =============================================================================
// Cohort-first Lattice endpoints
// =============================================================================

export async function getLatticeStatus(): Promise<LatticeStatusResponse> {
  return apiGet('/api/admin/cv/lattice-status');
}

export async function getCohortProjection(
  method: string,
  cohortId: number,
  params?: { perplexity?: number; n_neighbors?: number },
): Promise<EnrichedProjectionResponse> {
  const qs = new URLSearchParams({ cohort_id: String(cohortId) });
  if (params?.perplexity) qs.set('perplexity', String(params.perplexity));
  if (params?.n_neighbors) qs.set('n_neighbors', String(params.n_neighbors));
  return apiGet(`/api/admin/cv/projection/${method}?${qs}`);
}

// Open build: getOverlayProjection (POST /api/admin/cv/projection/{method}/overlay)
// was removed. Overlaying a second cohort re-scores raw feature matrices with
// the private per-fold model pickles, which the open backend does not carry.
