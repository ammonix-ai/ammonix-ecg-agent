import { PlugZap, RefreshCw, ServerCrash } from 'lucide-react';
import type { StatusProbe } from '@/services/statusService';

/**
 * Shown in place of the composer when the model cannot be reached.
 *
 * Two different situations, worded differently, because they have different
 * fixes: the backend itself is down, or the backend is up and its
 * OpenAI-compatible endpoint is not answering. Neither is a broken page —
 * Universe and Analyze keep working, and this panel says what to start.
 */
export function LlmOfflineNotice({
  probe,
  onRetry,
  retrying,
}: {
  probe: StatusProbe | null;
  onRetry: () => void;
  retrying: boolean;
}) {
  const backendDown = probe !== null && !probe.reachable;
  const model = probe?.status?.llmModel;
  const baseUrl = probe?.status?.llmBaseUrl;

  return (
    <div className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-1 p-4">
      <div className="flex items-start gap-2">
        {backendDown ? (
          <ServerCrash className="mt-0.5 w-4 h-4 shrink-0 text-text-secondary" aria-hidden />
        ) : (
          <PlugZap className="mt-0.5 w-4 h-4 shrink-0 text-text-secondary" aria-hidden />
        )}
        <div className="min-w-0 flex-1">
          <p className="text-body font-semibold text-text-primary">
            {backendDown ? 'Backend not reachable' : 'No language model connected'}
          </p>

          {backendDown ? (
            <p className="mt-1 text-caption leading-relaxed text-text-secondary">
              <code className="font-mono">GET /api/status</code> did not answer
              {probe?.error ? <> — {probe.error}</> : null}. Start the backend on port 8100:
              <br />
              <code className="mt-1 block font-mono text-text-primary">
                python -m uvicorn main:app --port 8100
              </code>
            </p>
          ) : (
            <>
              <p className="mt-1 text-caption leading-relaxed text-text-secondary">
                The backend is up and reports <code className="font-mono">llm: false</code>. Chat
                is disabled until an OpenAI-compatible endpoint answers. Universe and Analyze are
                unaffected — the ECG analysis does not need it.
              </p>
              <p className="mt-2 text-caption text-text-secondary">
                Point the backend at any OpenAI-compatible server (vLLM, Ollama, LM Studio) and
                restart it:
              </p>
              <pre className="mt-1 overflow-x-auto rounded-[var(--radius-sm)] bg-surface-2 p-2 font-mono text-caption text-text-primary">
{`OPENAI_BASE_URL=http://127.0.0.1:8001/v1
OPENAI_API_KEY=EMPTY
MODEL_NAME=<served model name>`}
              </pre>
              {(baseUrl || model) && (
                <p className="mt-2 font-mono text-caption text-text-muted">
                  configured: {baseUrl ?? 'no base url reported'}
                  {model ? ` · ${model}` : ''}
                </p>
              )}
            </>
          )}

          <button
            type="button"
            onClick={onRetry}
            disabled={retrying}
            className="mt-3 inline-flex items-center gap-1.5 rounded-[var(--radius-btn)] border border-[var(--border-default)] bg-surface-0 px-3 py-1.5 text-caption font-semibold text-text-primary hover:bg-surface-2 disabled:opacity-40"
          >
            <RefreshCw className={retrying ? 'w-3.5 h-3.5 animate-spin' : 'w-3.5 h-3.5'} aria-hidden />
            Check again
          </button>
        </div>
      </div>
    </div>
  );
}
