/**
 * Annotation store — patient favorites, pinning, and notes.
 * Persisted to localStorage across sessions.
 */
import { create } from 'zustand';

const STORAGE_KEY = 'patient-annotations';

interface Annotation {
  isFavorite: boolean;
  isPinned: boolean;
  notes: string[];
}

interface StoredData {
  [patientId: string]: Annotation;
}

function load(): StoredData {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* noop */ }
  return {};
}

function persist(data: StoredData) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch { /* noop */ }
}

const DEFAULT: Annotation = { isFavorite: false, isPinned: false, notes: [] };

interface AnnotationState {
  data: StoredData;

  toggleFavorite: (id: string) => void;
  togglePin: (id: string) => void;
  addNote: (id: string, text: string) => void;
  removeNote: (id: string, noteIndex: number) => void;

  isFavorite: (id: string) => boolean;
  isPinned: (id: string) => boolean;
  getAnnotation: (id: string) => Annotation;
  favoriteIds: () => string[];
  pinnedIds: () => string[];
  clearAll: () => void;
}

export const useAnnotationStore = create<AnnotationState>((set, get) => ({
  data: load(),

  toggleFavorite: (id) => {
    set((s) => {
      const entry = s.data[id] ?? { ...DEFAULT };
      const next = { ...s.data, [id]: { ...entry, isFavorite: !entry.isFavorite } };
      persist(next);
      return { data: next };
    });
  },

  togglePin: (id) => {
    set((s) => {
      const entry = s.data[id] ?? { ...DEFAULT };
      const next = { ...s.data, [id]: { ...entry, isPinned: !entry.isPinned } };
      persist(next);
      return { data: next };
    });
  },

  addNote: (id, text) => {
    set((s) => {
      const entry = s.data[id] ?? { ...DEFAULT };
      const next = { ...s.data, [id]: { ...entry, notes: [...entry.notes, text] } };
      persist(next);
      return { data: next };
    });
  },

  removeNote: (id, noteIndex) => {
    set((s) => {
      const entry = s.data[id];
      if (!entry) return s;
      const next = { ...s.data, [id]: { ...entry, notes: entry.notes.filter((_, i) => i !== noteIndex) } };
      persist(next);
      return { data: next };
    });
  },

  isFavorite: (id) => get().data[id]?.isFavorite ?? false,
  isPinned: (id) => get().data[id]?.isPinned ?? false,
  getAnnotation: (id) => get().data[id] ?? { ...DEFAULT },
  favoriteIds: () => Object.entries(get().data).filter(([, a]) => a.isFavorite).map(([id]) => id),
  pinnedIds: () => Object.entries(get().data).filter(([, a]) => a.isPinned).map(([id]) => id),

  clearAll: () => {
    persist({});
    set({ data: {} });
  },
}));
