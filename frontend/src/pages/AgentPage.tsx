import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ImageOff, Loader2, Send, Square } from 'lucide-react';
import { cn } from '@/design/cn';
import { useAnalyzeStore } from '@/stores/analyzeStore';
import { llmReady, useStatusStore } from '@/stores/statusStore';
import { streamAgentChat } from '@/services/agentChatService';
import { renderEcgPng } from '@/utils/renderEcgPng';
import { useEcgAnnotationStore } from '@/stores/ecgAnnotationStore';
import { WaveformPanel, ScoreKindTag } from '@/components/analyze';
import { LlmOfflineNotice } from '@/components/agent/LlmOfflineNotice';
import type { ChatMessage } from '@/types/analyze';

/**
 * AgentPage — chat about the recording on screen, with the trace next to it.
 *
 * The agent VERIFIES; it does not diagnose. The diagnosis comes from the
 * frozen classifier, the universe neighbours and the skills — all assembled
 * server-side. The 12 leads are drawn to a PNG and sent as `imageDataUrl` so
 * the model can check the classifier's leading features against the actual
 * trace. The starter prompts below are worded as verification requests for
 * that reason; none of them asks the model for an independent read.
 */
export default function AgentPage() {
  const selected = useAnalyzeStore((s) => s.selected);
  const adapted = useAnalyzeStore((s) => s.adapted);
  const result = useAnalyzeStore((s) => s.result);

  const probe = useStatusStore((s) => s.probe);
  const statusLoading = useStatusStore((s) => s.loading);
  const checkedAt = useStatusStore((s) => s.checkedAt);
  const refreshStatus = useStatusStore((s) => s.refresh);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [sendImage, setSendImage] = useState(true);
  const [imageSkipped, setImageSkipped] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (checkedAt === null) void refreshStatus();
  }, [checkedAt, refreshStatus]);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Follow the stream.
  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const ready = llmReady(probe);
  const hasTrace = Boolean(adapted);
  const recordingId = selected?.recordingId ?? null;

  const starters = useMemo(() => {
    const top = result?.topPrediction ?? result?.predictions?.[0] ?? null;
    const base = [
      'Which leads and intervals support the leading finding on this trace?',
      'Is there anything in the image that argues against the leading finding?',
    ];
    if (top) {
      base.unshift(`Check the evidence for "${top}" against what is visible in the trace.`);
    }
    return base;
  }, [result]);

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || streaming || !ready) return;

      const history: ChatMessage[] = [...messages, { role: 'user', content }];
      setMessages([...history, { role: 'assistant', content: '' }]);
      setDraft('');
      setStreaming(true);
      setStreamError(null);
      setImageSkipped(false);

      let imageDataUrl: string | null = null;
      if (sendImage && adapted) {
        // Include whatever the reader has drawn on the trace. Without this,
        // pointing at a wave and asking "what is this?" reaches a model that
        // cannot see what was pointed at.
        const marks = recordingId
          ? useEcgAnnotationStore.getState().forRecording(recordingId)
          : [];
        imageDataUrl = renderEcgPng(adapted.ecgData, { annotations: marks });
        if (!imageDataUrl) setImageSkipped(true);
      }

      abortRef.current = streamAgentChat(
        { messages: history, recordingId, imageDataUrl },
        {
          onDelta: (delta) => {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === 'assistant') {
                next[next.length - 1] = { role: 'assistant', content: last.content + delta };
              }
              return next;
            });
          },
          onComplete: () => {
            setStreaming(false);
            abortRef.current = null;
            // Drop an assistant turn the server never filled in.
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === 'assistant' && last.content === '') return prev.slice(0, -1);
              return prev;
            });
          },
          onError: (err) => {
            setStreaming(false);
            abortRef.current = null;
            setStreamError(err.message);
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === 'assistant' && last.content === '') return prev.slice(0, -1);
              return prev;
            });
          },
        },
      );
    },
    [adapted, messages, ready, recordingId, sendImage, streaming],
  );

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
  };

  return (
    <div className="flex h-full min-h-0 gap-4">
      {/* Trace */}
      <section className="flex min-w-0 flex-[3] flex-col gap-3">
        <header className="flex flex-wrap items-baseline justify-between gap-2">
          <h1 className="text-h2 text-text-primary">ECG Agent</h1>
          <Link to="/analyze" className="text-caption font-semibold text-brand hover:underline">
            {recordingId ? 'Change recording' : 'Pick a recording'}
          </Link>
        </header>

        <div className="min-h-0 flex-1 rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0 p-3">
          <WaveformPanel skipAnimation />
        </div>

        {/* What the agent is being asked to verify */}
        {result ? (
          <div className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-1 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-caption font-semibold uppercase tracking-wider text-text-muted">
                Findings for this recording
              </span>
              <ScoreKindTag kind={result.scoreKind} />
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {(result.predictions?.length ?? 0) === 0 ? (
                <span className="text-caption text-text-secondary">
                  Nothing cleared its threshold.
                </span>
              ) : (
                result.predictions.map((dx) => (
                  <span
                    key={dx}
                    className="rounded-[var(--radius-pill)] bg-brand-50 px-2 py-0.5 text-caption font-semibold text-brand"
                  >
                    {dx} {(result.probabilities?.[dx] ?? 0).toFixed(2)}
                  </span>
                ))
              )}
            </div>
          </div>
        ) : (
          <p className="text-caption text-text-secondary">
            No analysis loaded.{' '}
            <Link to="/analyze" className="font-semibold text-brand hover:underline">
              Run the analysis in Analyze
            </Link>{' '}
            first — this conversation checks those findings against the trace, it does not read
            the ECG on its own.
          </p>
        )}
      </section>

      {/* Conversation */}
      <section className="flex w-[26rem] shrink-0 flex-col gap-3 min-h-0">
        <div
          ref={transcriptRef}
          className="min-h-0 flex-1 space-y-3 overflow-y-auto rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0 p-3"
        >
          {messages.length === 0 ? (
            <div className="space-y-3">
              <p className="text-caption leading-relaxed text-text-secondary">
                Ask how the findings for this recording hold up against the trace. The
                conversation has the rendered 12-lead image, the probabilities, the thresholds,
                the leading features and the six nearest labelled recordings.
              </p>
              {ready && (
                <div className="space-y-1.5">
                  {starters.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => void send(s)}
                      className="w-full rounded-[var(--radius-btn)] border border-[var(--border-default)] bg-surface-1 px-3 py-2 text-left text-caption text-text-primary transition-colors hover:bg-surface-2"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            messages.map((m, i) => <Bubble key={i} message={m} />)
          )}

          {streaming && (
            <p className="flex items-center gap-1.5 text-caption text-text-muted">
              <Loader2 className="w-3 h-3 animate-spin" aria-hidden />
              streaming…
            </p>
          )}

          {streamError && (
            <p className="rounded-[var(--radius-sm)] border border-danger/40 bg-danger/5 p-2 text-caption text-text-secondary">
              <span className="font-semibold text-danger">Stream failed.</span> {streamError}
            </p>
          )}
        </div>

        {imageSkipped && (
          <p className="flex items-center gap-1.5 text-caption text-[color:var(--color-status-medium)]">
            <ImageOff className="w-3.5 h-3.5 shrink-0" aria-hidden />
            The trace could not be drawn to a PNG, so that message went without an image.
          </p>
        )}

        {ready ? (
          <Composer
            draft={draft}
            onDraft={setDraft}
            onSend={() => void send(draft)}
            onStop={stop}
            streaming={streaming}
            sendImage={sendImage}
            onSendImage={setSendImage}
            hasTrace={hasTrace}
          />
        ) : (
          <LlmOfflineNotice
            probe={probe}
            onRetry={() => void refreshStatus()}
            retrying={statusLoading}
          />
        )}
      </section>
    </div>
  );
}

function Bubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[92%] whitespace-pre-wrap rounded-[var(--radius-card)] px-3 py-2 text-body leading-relaxed',
          isUser ? 'bg-brand text-white' : 'bg-surface-1 text-text-primary',
        )}
      >
        {message.content}
      </div>
    </div>
  );
}

function Composer({
  draft,
  onDraft,
  onSend,
  onStop,
  streaming,
  sendImage,
  onSendImage,
  hasTrace,
}: {
  draft: string;
  onDraft: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  streaming: boolean;
  sendImage: boolean;
  onSendImage: (v: boolean) => void;
  hasTrace: boolean;
}) {
  return (
    <div className="rounded-[var(--radius-card)] border border-[var(--border-default)] bg-surface-0 p-2">
      <textarea
        value={draft}
        onChange={(e) => onDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSend();
          }
        }}
        rows={3}
        placeholder="Ask about a finding and where it shows on the trace…"
        aria-label="Message the ECG agent"
        className="w-full resize-none bg-transparent px-1 text-body text-text-primary outline-none placeholder:text-text-muted"
      />
      <div className="mt-1 flex items-center justify-between gap-2">
        <label
          className={cn(
            'flex items-center gap-1.5 text-caption',
            hasTrace ? 'text-text-secondary' : 'text-text-muted',
          )}
          title={
            hasTrace
              ? 'Draw the 12 leads to a PNG from the same samples and send it with the message. ' +
                'It is a redraw at the same calibration, not a screenshot of the panel.'
              : 'No trace loaded to draw.'
          }
        >
          <input
            type="checkbox"
            checked={sendImage && hasTrace}
            disabled={!hasTrace}
            onChange={(e) => onSendImage(e.target.checked)}
          />
          Attach the 12-lead as an image
        </label>
        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            className="inline-flex items-center gap-1.5 rounded-[var(--radius-btn)] border border-[var(--border-default)] px-3 py-1.5 text-caption font-semibold text-text-primary hover:bg-surface-2"
          >
            <Square className="w-3.5 h-3.5" aria-hidden />
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={onSend}
            disabled={draft.trim().length === 0}
            className="inline-flex items-center gap-1.5 rounded-[var(--radius-btn)] bg-brand px-3 py-1.5 text-caption font-semibold text-white hover:bg-brand-light disabled:opacity-40"
          >
            <Send className="w-3.5 h-3.5" aria-hidden />
            Send
          </button>
        )}
      </div>
    </div>
  );
}
