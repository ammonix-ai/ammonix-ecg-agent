import { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Search, Upload, X } from 'lucide-react';
import { cn } from '@/design/cn';
import { RECORD_SOURCES, type RecordSummary } from '@/types/analyze';
import { PAGE_SIZE, useAnalyzeStore } from '@/stores/analyzeStore';
import { uploadsAllowed, useStatusStore } from '@/stores/statusStore';
import { validateUpload } from '@/services/analyzeService';
import { DropZone } from '@/components/ui/DropZone';

/**
 * Browse the 10,876 shipped open-licensed recordings, or upload one.
 *
 * The counts and labels here are whatever GET /api/records returns. Nothing is
 * synthesised client-side: an empty list renders as an empty list.
 */
export function RecordBrowser() {
  const filters = useAnalyzeStore((s) => s.filters);
  const setFilter = useAnalyzeStore((s) => s.setFilter);
  const offset = useAnalyzeStore((s) => s.offset);
  const setOffset = useAnalyzeStore((s) => s.setOffset);
  const records = useAnalyzeStore((s) => s.records);
  const total = useAnalyzeStore((s) => s.total);
  const diagnoses = useAnalyzeStore((s) => s.diagnoses);
  const loading = useAnalyzeStore((s) => s.listLoading);
  const error = useAnalyzeStore((s) => s.listError);
  const loadRecords = useAnalyzeStore((s) => s.loadRecords);
  const selected = useAnalyzeStore((s) => s.selected);
  const selectRecord = useAnalyzeStore((s) => s.selectRecord);
  const uploadAndAnalyze = useAnalyzeStore((s) => s.uploadAndAnalyze);
  const analyzing = useAnalyzeStore((s) => s.analyzing);

  const [queryDraft, setQueryDraft] = useState(filters.q);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadProblem, setUploadProblem] = useState<string | null>(null);

  const probe = useStatusStore((s) => s.probe);
  const canUpload = uploadsAllowed(probe);

  // Debounce the id search so a keystroke is not a request.
  useEffect(() => {
    const timer = setTimeout(() => {
      if (queryDraft !== filters.q) setFilter({ q: queryDraft });
    }, 300);
    return () => clearTimeout(timer);
  }, [queryDraft, filters.q, setFilter]);

  useEffect(() => {
    void loadRecords();
  }, [loadRecords, filters.source, filters.diagnosis, filters.q, offset]);

  // Options come from the backend's corpus-wide vocabulary for the selected source.
  // Deriving them from `records` only ever offered the labels present in the current
  // 50-record page, so with "All sources" most of the corpus's diagnoses were
  // unreachable. Still no hardcoded class list: the vocabulary is whatever the
  // shipped gold labels contain.
  //
  // Fall back to the page-derived set when the backend does not send `diagnoses`
  // (it predates this field), so an un-restarted server degrades to the old
  // behaviour rather than to an empty list.
  const diagnosisOptions = useMemo(() => {
    const seen = new Set<string>(diagnoses);
    if (seen.size === 0) {
      for (const r of records) {
        for (const label of r.labels ?? []) seen.add(label);
      }
    }
    // Keep the active filter selectable even if it is outside the vocabulary,
    // so switching source never silently clears it.
    if (filters.diagnosis) seen.add(filters.diagnosis);
    return [...seen].sort((a, b) => a.localeCompare(b));
  }, [diagnoses, records, filters.diagnosis]);

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const handleUpload = (files: File[]) => {
    const problem = validateUpload(files);
    setUploadProblem(problem);
    if (problem) return;
    void uploadAndAnalyze(files);
  };

  return (
    <div className="flex flex-col h-full min-h-0 gap-3">
      {/* Filters */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0 rounded-[var(--radius-btn)] border border-[var(--border-default)] bg-surface-0 px-2 py-1.5">
            <Search className="w-3.5 h-3.5 text-text-muted shrink-0" aria-hidden />
            <input
              value={queryDraft}
              onChange={(e) => setQueryDraft(e.target.value)}
              placeholder="Recording id"
              aria-label="Search recording id"
              className="min-w-0 flex-1 bg-transparent text-body text-text-primary outline-none placeholder:text-text-muted"
            />
            {queryDraft && (
              <button
                type="button"
                onClick={() => setQueryDraft('')}
                aria-label="Clear search"
                className="text-text-muted hover:text-text-primary"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          {canUpload && (
          <button
            type="button"
            onClick={() => setShowUpload((v) => !v)}
            className={cn(
              'flex items-center gap-1 rounded-[var(--radius-btn)] border px-2 py-1.5 text-caption font-semibold transition-colors',
              showUpload
                ? 'border-brand bg-brand-50 text-brand'
                : 'border-[var(--border-default)] bg-surface-0 text-text-secondary hover:text-text-primary',
            )}
          >
            <Upload className="w-3.5 h-3.5" aria-hidden />
            Upload
          </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          <select
            value={filters.source}
            onChange={(e) => setFilter({ source: e.target.value })}
            aria-label="Filter by source"
            className="flex-1 min-w-0 rounded-[var(--radius-btn)] border border-[var(--border-default)] bg-surface-0 px-2 py-1.5 text-caption text-text-primary"
          >
            <option value="">All sources</option>
            {RECORD_SOURCES.map((src) => (
              <option key={src} value={src}>
                {src}
              </option>
            ))}
          </select>
          <select
            value={filters.diagnosis}
            onChange={(e) => setFilter({ diagnosis: e.target.value })}
            aria-label="Filter by diagnosis"
            className="flex-1 min-w-0 rounded-[var(--radius-btn)] border border-[var(--border-default)] bg-surface-0 px-2 py-1.5 text-caption text-text-primary"
          >
            <option value="">All diagnoses</option>
            {diagnosisOptions.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Upload panel */}
      {canUpload && showUpload && (
        <div className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-1 p-3">
          <p className="text-caption text-text-secondary mb-2">
            A WFDB pair (<span className="font-mono">.hea</span> +{' '}
            <span className="font-mono">.mat</span>/<span className="font-mono">.dat</span>) or a
            12-lead <span className="font-mono">.csv</span>. It is analysed exactly the same way —
            and unlike the shipped records, all five folds are out-of-sample for it.
          </p>
          <DropZone accept=".hea,.mat,.dat,.csv" multiple onDrop={handleUpload} />
          {uploadProblem && (
            <p className="mt-2 text-caption text-danger">{uploadProblem}</p>
          )}
          {analyzing && (
            <p className="mt-2 text-caption text-text-secondary animate-pulse">
              Tokenizing and classifying…
            </p>
          )}
        </div>
      )}

      {/* Result count */}
      <div className="flex items-center justify-between text-caption text-text-muted">
        <span>
          {loading
            ? 'Loading…'
            : `${total.toLocaleString()} recording${total === 1 ? '' : 's'}`}
        </span>
        {total > PAGE_SIZE && (
          <span className="flex items-center gap-1">
            <button
              type="button"
              disabled={offset === 0 || loading}
              onClick={() => setOffset(offset - PAGE_SIZE)}
              aria-label="Previous page"
              className="p-0.5 rounded hover:bg-surface-2 disabled:opacity-30"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="tabular-nums">
              {page} / {pageCount}
            </span>
            <button
              type="button"
              disabled={offset + PAGE_SIZE >= total || loading}
              onClick={() => setOffset(offset + PAGE_SIZE)}
              aria-label="Next page"
              className="p-0.5 rounded hover:bg-surface-2 disabled:opacity-30"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </span>
        )}
      </div>

      {/* List */}
      <div className="flex-1 min-h-0 overflow-y-auto rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0">
        {error ? (
          <div className="p-4">
            <p className="text-body text-danger font-semibold">Could not load recordings</p>
            <p className="text-caption text-text-secondary mt-1 break-words">{error}</p>
            <button
              type="button"
              onClick={() => void loadRecords()}
              className="mt-3 rounded-[var(--radius-btn)] bg-brand px-3 py-1.5 text-caption font-semibold text-white hover:bg-brand-light"
            >
              Retry
            </button>
          </div>
        ) : records.length === 0 && !loading ? (
          <p className="p-4 text-caption text-text-muted">
            No recordings match these filters.
          </p>
        ) : (
          <ul>
            {records.map((rec) => (
              <RecordRow
                key={rec.recordingId}
                record={rec}
                active={selected?.recordingId === rec.recordingId}
                onSelect={() => void selectRecord(rec)}
                matchDiagnosis={filters.diagnosis || undefined}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function RecordRow({
  record,
  active,
  onSelect,
  matchDiagnosis,
}: {
  record: RecordSummary;
  active: boolean;
  onSelect: () => void;
  /** The active diagnosis filter, so the row can show WHY this record matched. */
  matchDiagnosis?: string;
}) {
  const labels = record.labels ?? [];

  // `primary` is the severity-ranked pick of a record's labels, so filtering by
  // "abnormal q wave" could list a record whose primary is "wolff-parkinson-white"
  // and look like the wrong record came back. When a filter is active, lead with
  // the label that actually matched.
  const matched = matchDiagnosis
    ? labels.find((l) => l.toLowerCase() === matchDiagnosis.toLowerCase())
    : undefined;
  const headline = matched ?? record.primary ?? labels[0] ?? '—';
  const others = labels.filter((l) => l.toLowerCase() !== headline.toLowerCase()).length;
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        className={cn(
          'w-full text-left px-3 py-2 border-b border-[var(--border-subtle)] transition-colors',
          active ? 'bg-brand-50' : 'hover:bg-surface-1',
        )}
      >
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-mono text-body font-semibold text-text-primary truncate">
            {record.recordingId}
          </span>
          <span className="font-mono text-caption text-text-muted shrink-0">
            {record.displayId}
          </span>
        </div>
        <div className="flex items-center justify-between gap-2 mt-0.5">
          <span
            className="text-caption text-text-secondary truncate"
            title={labels.length ? labels.join(' · ') : undefined}
          >
            {headline}
            {others > 0 && (
              <span className="text-text-muted"> +{others}</span>
            )}
          </span>
          <span className="text-caption text-text-muted shrink-0">{record.source}</span>
        </div>
      </button>
    </li>
  );
}
