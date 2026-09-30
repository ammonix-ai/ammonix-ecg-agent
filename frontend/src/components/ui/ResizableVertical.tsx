import { useCallback, useEffect, useRef, useState } from 'react';
import { cn } from '@/design/cn';

/**
 * A panel the reader can drag taller or shorter, with the height remembered.
 *
 * The ECG needs this more than most panels: how much vertical room a 12-lead
 * grid wants depends on the screen, on whether the rhythm strip is shown, and
 * on how closely someone is looking at ST segments. A single fixed height is
 * wrong for somebody.
 *
 * Height is clamped to [minHeight, maxHeight] and persisted per `storageKey`,
 * so it survives a reload and a recording change.
 */
export interface ResizableVerticalProps {
  /** localStorage key. Distinct per panel so two panels do not share a height. */
  storageKey: string;
  defaultHeight: number;
  minHeight?: number;
  maxHeight?: number;
  children: React.ReactNode;
  className?: string;
  /** Accessible name for the drag handle. */
  label?: string;
}

const DEFAULT_MIN = 240;
const DEFAULT_MAX = 1800;
/** Arrow-key step, and a larger step with shift. */
const STEP = 24;
const STEP_LARGE = 96;

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

export function ResizableVertical({
  storageKey,
  defaultHeight,
  minHeight = DEFAULT_MIN,
  maxHeight = DEFAULT_MAX,
  children,
  className,
  label = 'Resize panel',
}: ResizableVerticalProps) {
  const [height, setHeight] = useState<number>(() => {
    if (typeof window === 'undefined') return defaultHeight;
    const stored = Number(window.localStorage.getItem(storageKey));
    return Number.isFinite(stored) && stored > 0
      ? clamp(stored, minHeight, maxHeight)
      : defaultHeight;
  });
  const [dragging, setDragging] = useState(false);

  // Persist lazily — writing on every pointermove would hammer localStorage.
  useEffect(() => {
    if (dragging) return;
    try {
      window.localStorage.setItem(storageKey, String(Math.round(height)));
    } catch {
      /* private mode / quota — the panel still works, it just will not persist */
    }
  }, [dragging, height, storageKey]);

  const dragState = useRef<{ startY: number; startH: number } | null>(null);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      dragState.current = { startY: e.clientY, startH: height };
      setDragging(true);
    },
    [height],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const s = dragState.current;
      if (!s) return;
      setHeight(clamp(s.startH + (e.clientY - s.startY), minHeight, maxHeight));
    },
    [minHeight, maxHeight],
  );

  const endDrag = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragState.current) return;
    dragState.current = null;
    setDragging(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, []);

  // Keyboard: the handle is focusable, so the panel is resizable without a mouse.
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const step = e.shiftKey ? STEP_LARGE : STEP;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setHeight((h) => clamp(h + step, minHeight, maxHeight));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setHeight((h) => clamp(h - step, minHeight, maxHeight));
      } else if (e.key === 'Home') {
        e.preventDefault();
        setHeight(defaultHeight);
      }
    },
    [minHeight, maxHeight, defaultHeight],
  );

  return (
    <>
      <div className={className} style={{ height }}>
        {children}
      </div>

      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label={label}
        aria-valuenow={Math.round(height)}
        aria-valuemin={minHeight}
        aria-valuemax={maxHeight}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={() => setHeight(defaultHeight)}
        onKeyDown={onKeyDown}
        title="Drag to resize · double-click to reset · arrow keys when focused"
        className={cn(
          'group -mt-1 flex h-3 shrink-0 cursor-ns-resize items-center justify-center',
          'touch-none select-none rounded focus:outline-none',
          'focus-visible:ring-2 focus-visible:ring-brand',
        )}
      >
        {/* A grip that is visible enough to find, quiet enough to ignore. */}
        <div
          className={cn(
            'h-1 w-16 rounded-full transition-colors',
            dragging
              ? 'bg-brand'
              : 'bg-[var(--border-default)] group-hover:bg-brand/60',
          )}
        />
      </div>
    </>
  );
}
