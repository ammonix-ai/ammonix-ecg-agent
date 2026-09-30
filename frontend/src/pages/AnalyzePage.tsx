import { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { AlertTriangle, Play, Loader2 } from 'lucide-react';
import { useAnalyzeStore } from '@/stores/analyzeStore';
import { uploadsAllowed, useStatusStore } from '@/stores/statusStore';
import { ResizableVertical } from '@/components/ui/ResizableVertical';
import {
  RecordBrowser,
  WaveformPanel,
  ClassificationPanel,
  FeatureContributions,
  UniversePlacement,
} from '@/components/analyze';

/**
 * AnalyzePage — pick one of the shipped recordings (or upload one), look at
 * the 12 leads, and run the frozen classifier over it.
 *
 * The pipeline behind the button is the real one:
 *   raw 12-lead WFDB → QPSI tokenizer → feat_order feature vector
 *   → 32 diagnoses x 5 frozen swarms → probabilities → placement
 *
 * Nothing on this page is simulated. If a call fails, the failure is what is
 * displayed.
 */
export default function AnalyzePage() {
  const selected = useAnalyzeStore((s) => s.selected);
  const origin = useAnalyzeStore((s) => s.origin);
  const result = useAnalyzeStore((s) => s.result);
  const analyzing = useAnalyzeStore((s) => s.analyzing);
  const analyzeError = useAnalyzeStore((s) => s.analyzeError);
  const analyzeSelected = useAnalyzeStore((s) => s.analyzeSelected);
  const selectByRecordingId = useAnalyzeStore((s) => s.selectByRecordingId);
  const canUpload = uploadsAllowed(useStatusStore((s) => s.probe));

  // Deep link from the universe: /analyze?recording=<id>
  const [searchParams] = useSearchParams();
  const deepLinkId = searchParams.get('recording');
  useEffect(() => {
    if (deepLinkId) void selectByRecordingId(deepLinkId);
  }, [deepLinkId, selectByRecordingId]);

  const isShippedRecord = origin === 'shipped';

  return (
    <div className="flex h-full min-h-0 gap-4">
      {/* Browser */}
      <aside className="flex w-[19rem] shrink-0 flex-col min-h-0">
        <h1 className="text-h2 text-text-primary">Analyze</h1>
        <p className="mb-3 text-caption text-text-secondary">
          {canUpload
            ? '10,876 open-licensed test recordings, or your own upload.'
            : '10,876 open-licensed test recordings.'}
        </p>
        <div className="min-h-0 flex-1">
          <RecordBrowser />
        </div>
      </aside>

      {/* Trace + results */}
      <div className="flex min-w-0 flex-1 flex-col gap-4 overflow-y-auto">
        {/* Default is tall enough for the full 12-lead grid plus the rhythm
            strip without scrolling; drag the handle to suit the screen. */}
        <ResizableVertical
          storageKey="analyze.waveformHeight"
          defaultHeight={660}
          minHeight={320}
          label="Resize the ECG panel"
          className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0 p-3"
        >
          <WaveformPanel />
        </ResizableVertical>

        {/* Run control */}
        {(selected || result || analyzing || analyzeError) && (
          <div className="flex flex-wrap items-center gap-3 rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-1 px-4 py-3">
            <button
              type="button"
              onClick={() => void analyzeSelected()}
              disabled={!selected || analyzing}
              className="inline-flex items-center gap-2 rounded-[var(--radius-btn)] bg-brand px-4 py-2 text-body font-semibold text-white transition-colors hover:bg-brand-light disabled:opacity-40"
            >
              {analyzing ? (
                <Loader2 className="w-4 h-4 animate-spin" aria-hidden />
              ) : (
                <Play className="w-4 h-4" aria-hidden />
              )}
              {analyzing ? 'Running…' : result ? 'Run again' : 'Run the analysis'}
            </button>
            <p className="text-caption text-text-secondary">
              Tokenizes the raw waveform and scores it against all 32 diagnoses. Expect a few
              seconds; without the optional native accelerator the first run can take ~25 s.
            </p>
          </div>
        )}

        {analyzeError && (
          <div className="flex items-start gap-2 rounded-[var(--radius-card)] border border-danger/40 bg-danger/5 p-4">
            <AlertTriangle className="mt-0.5 w-4 h-4 shrink-0 text-danger" aria-hidden />
            <div className="min-w-0">
              <p className="text-body font-semibold text-text-primary">Analysis failed</p>
              <p className="mt-1 break-words text-caption text-text-secondary">{analyzeError}</p>
            </div>
          </div>
        )}

        {result && (
          <div className="grid gap-4 xl:grid-cols-2">
            <div className="flex flex-col gap-4">
              <ClassificationPanel result={result} isShippedRecord={isShippedRecord} />
            </div>
            <div className="flex flex-col gap-4">
              <UniversePlacement result={result} selfDisplayId={selected?.displayId ?? null} />
              <FeatureContributions result={result} />
            </div>
          </div>
        )}

        {!result && !analyzing && !selected && (
          <p className="px-1 text-caption text-text-muted">
            Nothing is classified until you ask for it — selecting a recording only reads its
            waveform.
          </p>
        )}
      </div>
    </div>
  );
}
