/**
 * Wire shapes for the Analyze + ECG Agent endpoints.
 *
 * Mirrors `docs/api-contract.md` §2 and §3. Fields the contract lists are
 * required; fields the backend's inference runtime returns *in addition* to
 * the contract (`thresholds`, `foldProbabilities`, `modelVersion`, the PCA /
 * UMAP placement flags) are optional here, so the UI works against either a
 * strict-contract router or the fuller payload without pretending a missing
 * field is a zero.
 */

// ── GET /api/records ───────────────────────────────────────────────────

/** The five open-licensed PhysioNet sources shipped in `_staging/traces`. */
export const RECORD_SOURCES = ['PTB-XL', 'CPSC', 'CPSC-Extra', 'Georgia', 'Chapman'] as const;
export type RecordSource = (typeof RECORD_SOURCES)[number];

export interface RecordSummary {
  recordingId: string;
  displayId: string;
  source: string;
  labels?: string[];
  primary?: string | null;
  /** Score stored in the published universe for this row. */
  storedTopPrediction?: string | null;
  storedMaxScore?: number | null;
}

export interface RecordListResponse {
  total: number;
  records: RecordSummary[];
  /**
   * Every gold label available under the current `source`, for the diagnosis
   * filter's suggestions. Corpus-wide, so it does not depend on which 50-record
   * page happens to be loaded.
   */
  diagnoses?: string[];
}

export interface RecordQuery {
  source?: string;
  diagnosis?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

// ── GET /api/records/{id}/signal ───────────────────────────────────────

export interface SignalLead {
  name: string;
  samples: number[];
}

export interface RecordSignal {
  recordingId: string;
  samplingRate: number;
  durationSec: number;
  units: string;
  leads: SignalLead[];
}

// ── POST /api/analyze ──────────────────────────────────────────────────

export interface FeatureContribution {
  name: string;
  value: number;
  /** Signed log-odds contribution, mean SHAP over the five folds. */
  contribution: number;
}

export interface UniverseNeighbor {
  displayId: string;
  distance: number;
  primary?: string | null;
}

export interface ProjectionPoint {
  pca?: number[] | null;
  umap?: number[] | null;
  tsne?: number[] | null;
}

/**
 * The score already stored in the published universe for a shipped record.
 * `scoreKind` is carried verbatim from the backend — the UI never assumes it
 * matches the live `scoreKind`, because those are different quantities.
 */
export interface StoredScore {
  topPrediction?: string | null;
  maxScore?: number | null;
  scoreKind?: string | null;
}

export interface AnalyzeResult {
  recordingId: string;
  /** `ensemble_5fold` | `oof_single_fold` | anything the backend reports. */
  scoreKind: string;
  featureCount: number;
  probabilities: Record<string, number>;
  predictions: string[];
  topFeatures: FeatureContribution[];
  projection: ProjectionPoint;
  /** `knn_approx` when the point was placed by neighbour centroid, not transform. */
  tsnePlacement?: string;
  pcaPlacement?: string;
  pcaResidualRms?: number;
  umapPlacement?: string;
  umapNote?: string;
  neighbors: UniverseNeighbor[];
  stored?: StoredScore | null;
  /**
   * The analysed waveform, returned for UPLOADS only.
   *
   * Uploaded files are not retained after the request, so there is no
   * `GET /api/records/{id}/signal` to read back — without this the reader
   * cannot see the trace they just submitted. Shipped records omit it and use
   * the endpoint instead, since it is ~380 KB per recording.
   */
  signal?: RecordSignal | null;

  // Present on the fuller runtime payload; absent on a strict-contract router.
  thresholds?: Record<string, number>;
  thresholdKind?: string;
  topPrediction?: string;
  maxScore?: number;
  modelVersion?: string;
  /** Per-fold probabilities, five entries per diagnosis. */
  foldProbabilities?: Record<string, number[]>;
  /** True when the analyzed waveform came from an upload, not the shipped set. */
  uploaded?: boolean;
}

// ── GET /api/status ────────────────────────────────────────────────────

/**
 * Everything is optional: the contract only pins `llm`, and the UI must not
 * invent a value for a field the backend chose not to report.
 */
export interface ApiStatus {
  status?: string;
  /** False when the OpenAI-compatible endpoint is unreachable. */
  llm?: boolean;
  llmModel?: string | null;
  llmBaseUrl?: string | null;
  classifier?: boolean;
  /** The backend reports each data package it has installed. */
  universe?: boolean | { ready?: boolean; points?: number | null } | null;
  model?: { packagePresent?: boolean; loaded?: boolean } | null;
  traces?: { available?: boolean; records?: number | null } | null;
  modelVersion?: string | null;
  featureCount?: number | null;
  recordCount?: number | null;
  universePoints?: number | null;
  /** Absent on older backends, which always accept uploads. */
  analyze?: { uploadsEnabled?: boolean } | null;
}

// ── POST /api/chat/stream ──────────────────────────────────────────────

export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}
