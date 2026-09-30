import { apiGet } from '@/api/client'
import type {
  ProjectionMethod,
  EnrichedProjectionResponse,
  CVEnrichedProjectionResponse,
  FoldMode,
} from '@/types/projection'

/**
 * Projection Service — t-SNE, UMAP and PCA projections for the Universe view.
 *
 * Endpoints:
 * - GET /api/projections/{method}/enriched     (with predictions & tribes)
 * - GET /api/projections/cv/{method}/enriched  (CV fold-enriched)
 * - GET /api/admin/cv/source-safe-roc          (per-diagnosis ROC on the universe)
 *
 * Open build: everything else this module used to reach is gone with the pages
 * that called it — the base/legacy projection endpoints, the projection cache
 * listing, the UMAP transform, and the per-patient source-safe inference /
 * ECG-image / agent-report calls, none of which the open backend serves.
 */

export interface ProjectionParams {
  cohorts?: string
  perplexity?: number
  n_neighbors?: number
  force_recompute?: boolean
  fold_mode?: string
}

/** Fetch enriched projection (includes predictions, tribes, accuracy). */
export async function getEnrichedProjection(
  method: ProjectionMethod = 'tsne',
  params?: ProjectionParams,
): Promise<EnrichedProjectionResponse> {
  const query = new URLSearchParams()
  if (params?.cohorts) query.set('cohorts', params.cohorts)
  if (params?.perplexity !== undefined) query.set('perplexity', String(params.perplexity))
  if (params?.n_neighbors !== undefined) query.set('n_neighbors', String(params.n_neighbors))
  if (params?.force_recompute) query.set('force_recompute', 'true')
  if (params?.fold_mode) query.set('fold_mode', params.fold_mode)
  const qs = query.toString()
  return apiGet<EnrichedProjectionResponse>(
    `/api/projections/${method}/enriched${qs ? `?${qs}` : ''}`,
  )
}

// =============================================================================
// CV projection endpoints
// =============================================================================

export interface CVProjectionParams {
  method?: ProjectionMethod
  foldMode?: FoldMode
  perplexity?: number
  n_neighbors?: number
}

/** Fetch CV-enriched projection with fold/holdout support. */
export async function getCVProjection(
  params: CVProjectionParams = {},
): Promise<CVEnrichedProjectionResponse> {
  const method = params.method || 'umap'
  const query = new URLSearchParams()
  if (params.foldMode) query.set('fold_mode', params.foldMode)
  if (params.perplexity !== undefined) query.set('perplexity', String(params.perplexity))
  if (params.n_neighbors !== undefined) query.set('n_neighbors', String(params.n_neighbors))
  const qs = query.toString()
  return apiGet<CVEnrichedProjectionResponse>(
    `/api/projections/cv/${method}/enriched${qs ? `?${qs}` : ''}`,
  )
}

// =============================================================================
// Source-safe universe — per-diagnosis ROC (computed ON the displayed universe)
// =============================================================================

/** One ROC point. */
export interface SourceSafeROCPoint {
  fpr: number
  tpr: number
}

/**
 * ROC for one diagnosis over the displayed source-safe universe.
 * `auroc` is the area under THIS curve over THIS population — deliberately
 * distinct from the model's held-out numbers (`heldOutFinalAuroc` /
 * `heldOutPrimaryAuroc`), which are returned for reference so the UI can show
 * both without conflating them.
 */
export interface SourceSafeROC {
  diagnosis: string
  points: SourceSafeROCPoint[]
  auroc: number
  threshold: number
  operatingFpr: number | null
  operatingTpr: number | null
  nPositive: number
  nNegative: number
  heldOutFinalAuroc: number | null
  heldOutPrimaryAuroc: number | null
}

/** Fetch the on-universe ROC curve for a single source-safe diagnosis.
 * `diagnosis` goes in the query string (not the path) — several class names
 * contain a '/' that breaks path-param routing. */
export async function getSourceSafeROC(diagnosis: string): Promise<SourceSafeROC> {
  return apiGet<SourceSafeROC>(`/api/admin/cv/source-safe-roc?diagnosis=${encodeURIComponent(diagnosis)}`)
}
