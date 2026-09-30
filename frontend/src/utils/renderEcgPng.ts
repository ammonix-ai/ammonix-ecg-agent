/**
 * Render a 12-lead ECG to a PNG data URL, drawn from the samples on a canvas.
 *
 * This is what gets attached to a chat message as `imageDataUrl`, so the model
 * can look at the trace the classifier scored.
 *
 * It draws from the same sample arrays as the on-screen renderer, in the same
 * 2x6 lead order, at the same calibration — but it is a second renderer, not a
 * screenshot: the pixels are not identical to the SVG on screen.
 *
 * Marks the reader has drawn ARE included, placed on the lead they belong to.
 * They are stored as (lead, seconds, millivolts) rather than pixels, so this
 * renderer can place them at its own scale even though they were drawn in a
 * zoomed single-lead viewport. Without this, pointing at a wave and asking
 * "what is this?" reached a model that could not see what was pointed at.
 *
 * Why not rasterise the DOM: `html-to-image` never resolves against this app's
 * document (verified in-browser — `toCanvas` timed out at 15 s on the ECG and
 * at 8 s on the page header alone), which left the composer stuck on "sending"
 * with no request ever issued. A direct canvas draw is synchronous, has no
 * dependency, and cannot hang.
 *
 * Grid semantics are the clinical ones and are printed into the image:
 * a large box is 0.2 s wide and 0.5 mV tall, a small box 0.04 s and 0.1 mV.
 */

import { LEAD_GRID_2x6, ECG_GRID_COLORS } from '@/types/ecg';
import type { ECGData, SampleArray } from '@/types/ecg';
import type { EcgAnnotation } from '@/stores/ecgAnnotationStore';

export interface EcgPngOptions {
  width?: number;
  height?: number;
  /** Seconds of signal to draw, from the start. */
  maxSeconds?: number;
  /** Vertical scale in pixels per millivolt. */
  pixelsPerMv?: number;
  /**
   * Marks the reader drew on the trace, in (lead, seconds, millivolts).
   *
   * These are drawn onto the matching lead so that pointing at a wave and
   * asking "what is this?" reaches a model that can actually see what was
   * pointed at. Stored in signal units rather than pixels precisely so they
   * can be re-placed at this renderer's scale — see `ecgAnnotationStore`.
   */
  annotations?: EcgAnnotation[];
}

const DEFAULTS = {
  width: 1200,
  height: 900,
  maxSeconds: 10,
  pixelsPerMv: 30,
};

const LABEL_GUTTER = 34;
const HEADER_H = 26;
const FOOTER_H = 18;

function sampleAt(samples: SampleArray, i: number): number {
  const v = samples[i];
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** Returns a `data:image/png;base64,...` URL, or null if canvas is unavailable. */
export function renderEcgPng(ecg: ECGData, options: EcgPngOptions = {}): string | null {
  const width = options.width ?? DEFAULTS.width;
  const height = options.height ?? DEFAULTS.height;
  const maxSeconds = options.maxSeconds ?? DEFAULTS.maxSeconds;
  const pxPerMv = options.pixelsPerMv ?? DEFAULTS.pixelsPerMv;
  const annotations = options.annotations ?? [];

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const fs = ecg.samplingRate > 0 ? ecg.samplingRate : 500;
  const totalSec = ecg.numSamples / fs;
  const seconds = Math.min(maxSeconds, totalSec) || maxSeconds;

  const rows = LEAD_GRID_2x6.length;
  const cols = LEAD_GRID_2x6[0]?.length ?? 2;
  const gridTop = HEADER_H;
  const gridH = height - HEADER_H - FOOTER_H;
  const cellW = (width - LABEL_GUTTER * cols) / cols;
  const cellH = gridH / rows;

  const pxPerSec = cellW / seconds;
  const majorX = 0.2 * pxPerSec; // 0.2 s
  const minorX = 0.04 * pxPerSec;
  const majorY = 0.5 * pxPerMv; // 0.5 mV
  const minorY = 0.1 * pxPerMv;

  ctx.fillStyle = ECG_GRID_COLORS.background;
  ctx.fillRect(0, 0, width, height);

  // ── header ──
  ctx.fillStyle = '#1A1A2E';
  ctx.font = 'bold 14px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.fillText(ecg.patientId, 8, HEADER_H / 2);
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillStyle = '#6C757D';
  ctx.fillText(
    `${fs} Hz · ${seconds.toFixed(1)} s shown · large box 0.2 s x 0.5 mV`,
    12 + ctx.measureText(ecg.patientId).width + 40,
    HEADER_H / 2,
  );

  for (let r = 0; r < rows; r += 1) {
    const rowLeads = LEAD_GRID_2x6[r];
    if (!rowLeads) continue;
    for (let c = 0; c < cols; c += 1) {
      const lead = rowLeads[c];
      if (!lead) continue;

      const x0 = c * (cellW + LABEL_GUTTER) + LABEL_GUTTER;
      const y0 = gridTop + r * cellH;
      const midY = y0 + cellH / 2;

      // grid
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, y0, cellW, cellH);
      ctx.clip();

      if (minorX >= 3) {
        ctx.strokeStyle = ECG_GRID_COLORS.minor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = x0; x <= x0 + cellW; x += minorX) {
          ctx.moveTo(Math.round(x) + 0.5, y0);
          ctx.lineTo(Math.round(x) + 0.5, y0 + cellH);
        }
        for (let y = midY; y <= y0 + cellH; y += minorY) {
          ctx.moveTo(x0, Math.round(y) + 0.5);
          ctx.lineTo(x0 + cellW, Math.round(y) + 0.5);
        }
        for (let y = midY; y >= y0; y -= minorY) {
          ctx.moveTo(x0, Math.round(y) + 0.5);
          ctx.lineTo(x0 + cellW, Math.round(y) + 0.5);
        }
        ctx.stroke();
      }

      ctx.strokeStyle = ECG_GRID_COLORS.major;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = x0; x <= x0 + cellW; x += majorX) {
        ctx.moveTo(Math.round(x) + 0.5, y0);
        ctx.lineTo(Math.round(x) + 0.5, y0 + cellH);
      }
      for (let y = midY; y <= y0 + cellH; y += majorY) {
        ctx.moveTo(x0, Math.round(y) + 0.5);
        ctx.lineTo(x0 + cellW, Math.round(y) + 0.5);
      }
      for (let y = midY; y >= y0; y -= majorY) {
        ctx.moveTo(x0, Math.round(y) + 0.5);
        ctx.lineTo(x0 + cellW, Math.round(y) + 0.5);
      }
      ctx.stroke();

      // trace
      const samples = ecg.leads[lead];
      if (samples && samples.length > 0) {
        const n = Math.min(samples.length, Math.round(seconds * fs));
        ctx.strokeStyle = ECG_GRID_COLORS.trace;
        ctx.lineWidth = ECG_GRID_COLORS.traceWidth;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        // One polyline point per output pixel column: the min/max pair per
        // column would double the stroke count for no gain at this size.
        const step = Math.max(1, Math.floor(n / cellW));
        for (let i = 0; i < n; i += step) {
          const x = x0 + (i / fs) * pxPerSec;
          const y = midY - sampleAt(samples, i) * pxPerMv;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      } else {
        ctx.fillStyle = '#ADB5BD';
        ctx.font = '11px system-ui, sans-serif';
        ctx.fillText('no data', x0 + 6, midY);
      }
      ctx.restore();

      // lead label in the gutter
      ctx.fillStyle = '#1A1A2E';
      ctx.font = 'bold 12px system-ui, sans-serif';
      ctx.fillText(lead, x0 - LABEL_GUTTER + 4, y0 + 12);

      // ── reader's marks on this lead ──
      // Drawn after the clip is released so a label near the cell edge is not
      // sliced off. Anything outside the drawn time window is skipped rather
      // than clamped, so a mark never appears at a time it was not made.
      for (const ann of annotations) {
        if (ann.lead !== lead) continue;
        if (ann.tSeconds < 0 || ann.tSeconds > seconds) continue;

        const ax = x0 + ann.tSeconds * pxPerSec;
        const ay = midY - ann.mV * pxPerMv;
        const colour = ann.color || '#D6336C';

        if (ann.kind === 'stroke' && ann.points?.length) {
          ctx.strokeStyle = colour;
          ctx.lineWidth = 2;
          ctx.lineJoin = 'round';
          ctx.beginPath();
          ann.points.forEach((p, i) => {
            const px = x0 + p.tSeconds * pxPerSec;
            const py = midY - p.mV * pxPerMv;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.stroke();
        } else {
          // Ring rather than a filled blob, so the waveform stays readable
          // underneath the mark.
          ctx.strokeStyle = colour;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(ax, ay, 5, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (ann.label) {
          ctx.font = 'bold 11px system-ui, sans-serif';
          const textW = ctx.measureText(ann.label).width;
          // Flip the callout to the left when it would run past the cell edge.
          const flip = ax + 10 + textW + 8 > x0 + cellW;
          const boxX = flip ? ax - 10 - textW - 8 : ax + 10;
          const boxY = Math.max(y0 + 2, Math.min(ay - 20, y0 + cellH - 18));

          ctx.strokeStyle = colour;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(flip ? boxX + textW + 8 : boxX, boxY + 8);
          ctx.stroke();

          ctx.fillStyle = colour;
          ctx.fillRect(boxX, boxY, textW + 8, 16);
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(ann.label, boxX + 4, boxY + 12);
        }
      }
    }
  }

  // ── footer: calibration statement ──
  ctx.fillStyle = '#6C757D';
  ctx.font = '11px system-ui, sans-serif';
  const drawn = annotations.filter((a) => a.tSeconds >= 0 && a.tSeconds <= seconds).length;
  ctx.fillText(
    `${pxPerMv} px/mV vertical · ${pxPerSec.toFixed(1)} px/s horizontal · drawn from the raw samples, not a screenshot`
      + (drawn ? ` · ${drawn} mark${drawn === 1 ? '' : 's'} added by the reader` : ''),
    8,
    height - FOOTER_H / 2,
  );

  try {
    return canvas.toDataURL('image/png');
  } catch {
    return null;
  }
}
