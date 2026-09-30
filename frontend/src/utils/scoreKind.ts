/**
 * Score-kind vocabulary.
 *
 * `ensemble_5fold` and `oof_single_fold` are DIFFERENT QUANTITIES and the UI
 * must never blend them into one number or let one stand in for the other.
 * Every place a probability is rendered, the kind that produced it is rendered
 * next to it, from this table.
 *
 *   ensemble_5fold   mean over all five frozen swarms
 *   oof_single_fold  the one fold that held that record out
 *
 * A caveat the Analyze page has to carry: for a recording that is already in
 * the published universe, the five-fold mean includes folds that trained on
 * it, so the number is NOT out-of-sample performance. For a genuine upload all
 * five folds are out-of-sample and the same number is honest. `foldSpread()`
 * exists so the per-fold disagreement can be shown rather than described.
 */

export const ENSEMBLE_5FOLD = 'ensemble_5fold';
export const OOF_SINGLE_FOLD = 'oof_single_fold';

export interface ScoreKindInfo {
  /** Short label rendered inline next to a number. */
  label: string;
  /** One sentence saying exactly what the number is. */
  meaning: string;
}

const KINDS: Record<string, ScoreKindInfo> = {
  [ENSEMBLE_5FOLD]: {
    label: '5-fold ensemble',
    meaning: 'Mean probability over all five frozen folds.',
  },
  [OOF_SINGLE_FOLD]: {
    label: 'held-out fold',
    meaning: 'Probability from the single fold that held this record out of training.',
  },
};

/** Never invents a description: an unknown kind is echoed verbatim. */
export function scoreKindInfo(kind: string | null | undefined): ScoreKindInfo {
  if (!kind) {
    return { label: 'unlabelled', meaning: 'The backend did not report how this score was produced.' };
  }
  return KINDS[kind] ?? { label: kind, meaning: `Score kind reported by the backend as "${kind}".` };
}

/**
 * The caveat for a score computed on a recording that is part of the published
 * universe. Returns null when it does not apply (an upload, or a kind that is
 * already out-of-sample by construction).
 */
export function trainingOverlapCaveat(
  kind: string | null | undefined,
  isShippedRecord: boolean,
): string | null {
  if (!isShippedRecord) return null;
  if (kind !== ENSEMBLE_5FOLD) return null;
  return (
    'This recording is in the published universe, so most of the five folds saw it during ' +
    'training. The mean below is therefore not out-of-sample performance. Upload a recording ' +
    'that has never been seen to get a score where all five folds are out-of-sample.'
  );
}

export interface FoldSpread {
  values: number[];
  min: number;
  max: number;
  mean: number;
  /** max − min. Large spread on a shipped record usually means one fold held it out. */
  range: number;
}

export function foldSpread(values: number[] | undefined | null): FoldSpread | null {
  if (!values || values.length === 0) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return { values, min, max, mean, range: max - min };
}

/**
 * Placement wording. `knn_approx` is an approximation and is always named as
 * one — an approximated coordinate is never presented as exact.
 */
export function placementInfo(placement: string | null | undefined): ScoreKindInfo {
  switch (placement) {
    case 'knn_approx':
      return {
        label: 'approximate',
        meaning:
          'No out-of-sample transform exists for this projection, so the point sits at the ' +
          'distance-weighted centroid of its 6 nearest neighbours in probability space. ' +
          'It is an approximation, not an exact coordinate.',
      };
    case 'transform':
      return {
        label: 'exact',
        meaning: "Placed by the fitted projection model's transform().",
      };
    case 'linear_least_squares':
      return {
        label: 're-derived linear map',
        meaning:
          'PCA is linear, so the projection was recovered by least squares from the published ' +
          'probabilities onto the published coordinates. A re-derivation, not a refit.',
      };
    case 'exact':
      return { label: 'exact', meaning: 'Exact coordinate from the published embedding.' };
    default:
      if (!placement) {
        return { label: 'unreported', meaning: 'The backend did not say how this point was placed.' };
      }
      return { label: placement, meaning: `Placement reported by the backend as "${placement}".` };
  }
}

export function isApproximate(placement: string | null | undefined): boolean {
  return placement === 'knn_approx';
}
