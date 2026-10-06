import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderEcgPng } from './renderEcgPng';
import { ECG_GRID_COLORS, type ECGData } from '@/types/ecg';

afterEach(() => vi.restoreAllMocks());

describe('ECG image waveform fidelity', () => {
  it('retains positive and negative single-sample spikes between old stride positions', () => {
    let path: number[][] = [];
    const traces: number[][][] = [];
    const ctx = {
      strokeStyle: '', lineWidth: 1, lineJoin: '', fillStyle: '', font: '', textBaseline: '',
      fillRect: vi.fn(), fillText: vi.fn(), save: vi.fn(), restore: vi.fn(),
      rect: vi.fn(), clip: vi.fn(), measureText: () => ({ width: 40 }),
      beginPath: () => { path = []; },
      moveTo: (x: number, y: number) => { path.push([x, y]); },
      lineTo: (x: number, y: number) => { path.push([x, y]); },
      stroke: () => { if (ctx.strokeStyle === ECG_GRID_COLORS.trace) traces.push([...path]); },
    };
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/png;base64,test');
    const samples = new Float32Array(5000);
    samples[3] = 1;
    samples[5] = -1;
    samples[4999] = 2;
    const ecg = { patientId: 'synthetic', samplingRate: 500, numSamples: 5000,
      leads: { I: samples } } as ECGData;
    renderEcgPng(ecg);
    const trace = traces[0]!;
    const baseline = trace[0]![1]!;
    // Check the actual drawn extrema and their timing, not merely loop length.
    for (const [sample, amplitude] of [[3, 1], [5, -1], [4999, 2]]) {
      const expectedX = 34 + sample! / 500 * 56.6;
      expect(trace.some(([x, y]) => Math.abs(x! - expectedX) < 1e-8 &&
        Math.abs(y! - (baseline - amplitude! * 30)) < 1e-8)).toBe(true);
    }
  });
});
