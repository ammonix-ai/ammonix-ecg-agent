import { useState } from 'react';
import { Activity, AlertTriangle } from 'lucide-react';
import { TwelveLeadView } from '@/components/ecg/twelve-lead/TwelveLeadView';
import { ExpandedLeadView } from '@/components/ecg/expanded/ExpandedLeadView';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { useAnalyzeStore } from '@/stores/analyzeStore';
import type { LeadName } from '@/types/ecg';

/**
 * The 12-lead trace for the current selection.
 *
 * Reuses the clinical renderer (IEC 60601-2-25 grid, calibration pulse,
 * rhythm strip, per-lead expansion) rather than drawing a second, prettier
 * waveform that would not be measurable.
 *
 * The image the ECG Agent sends to the model is drawn separately from the same
 * samples (see `utils/renderEcgPng.ts`) — it is not a screenshot of this panel.
 */
export function WaveformPanel({ skipAnimation = false }: { skipAnimation?: boolean }) {
  const selected = useAnalyzeStore((s) => s.selected);
  const origin = useAnalyzeStore((s) => s.origin);
  const uploadNames = useAnalyzeStore((s) => s.uploadNames);
  const adapted = useAnalyzeStore((s) => s.adapted);
  const loading = useAnalyzeStore((s) => s.signalLoading);
  const error = useAnalyzeStore((s) => s.signalError);
  const predictions = useAnalyzeStore((s) => s.result?.predictions);

  const [expandedLead, setExpandedLead] = useState<LeadName | null>(null);

  if (loading) {
    return <LoadingSpinner message="Reading the waveform…" className="h-full" />;
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        <AlertTriangle className="w-6 h-6 text-danger" aria-hidden />
        <p className="text-body font-semibold text-text-primary">Waveform unavailable</p>
        <p className="text-caption text-text-secondary max-w-md break-words">{error}</p>
      </div>
    );
  }

  if (!adapted) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        <Activity className="w-6 h-6 text-text-muted" aria-hidden />
        {origin === 'upload' ? (
          <>
            <p className="text-body font-semibold text-text-primary">
              Uploaded recording — waveform not returned
            </p>
            <p className="text-caption text-text-secondary max-w-md">
              The analysis ran, but this response carried no samples to draw. An upload is not
              retained after the request, so the waveform has to travel with the result — if you
              are seeing this, the server did not send it.
              {uploadNames.length > 0 && (
                <>
                  {' '}
                  Files: <span className="font-mono">{uploadNames.join(', ')}</span>.
                </>
              )}
            </p>
          </>
        ) : (
          <>
            <p className="text-body font-semibold text-text-primary">No recording selected</p>
            <p className="text-caption text-text-secondary">
              Pick one from the list to draw its 12 leads.
            </p>
          </>
        )}
      </div>
    );
  }

  const { ecgData, missingLeads, unknownLeads } = adapted;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-1 pb-2">
        <span className="font-mono text-body font-semibold text-text-primary">
          {ecgData.patientId}
        </span>
        {selected && (
          <span className="font-mono text-caption text-text-muted">{selected.displayId}</span>
        )}
        <span className="text-caption text-text-secondary">
          {ecgData.samplingRate} Hz · {(ecgData.durationSeconds ?? 0).toFixed(1)} s ·{' '}
          {ecgData.numSamples.toLocaleString()} samples
        </span>
        {selected?.source && (
          <span className="text-caption text-text-muted">{selected.source}</span>
        )}
      </div>

      {(missingLeads.length > 0 || unknownLeads.length > 0) && (
        <p className="mb-2 flex items-start gap-1.5 px-1 text-caption text-[color:var(--color-status-medium)]">
          <AlertTriangle className="mt-0.5 w-3.5 h-3.5 shrink-0" aria-hidden />
          <span>
            {missingLeads.length > 0 && (
              <>Not present in the payload: {missingLeads.join(', ')}. Those frames stay empty.</>
            )}
            {unknownLeads.length > 0 && (
              <> Unrecognised lead names: {unknownLeads.join(', ')}.</>
            )}
          </span>
        </p>
      )}

      <div className="flex-1 min-h-0">
        {expandedLead ? (
          <ExpandedLeadView
            ecgData={ecgData}
            leadName={expandedLead}
            onBack={() => setExpandedLead(null)}
            onLeadChange={(lead) => setExpandedLead(lead)}
            topDiagnoses={predictions ?? []}
            recordingId={selected?.recordingId}
            className="h-full"
          />
        ) : (
          <TwelveLeadView
            ecgData={ecgData}
            onExpandLead={(lead) => setExpandedLead(lead)}
            skipAnimation={skipAnimation}
            className="h-full"
          />
        )}
      </div>
    </div>
  );
}
