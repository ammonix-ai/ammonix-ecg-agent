import { useEffect } from 'react'
import { X, Keyboard } from 'lucide-react'

interface KeyboardShortcutsOverlayProps {
  open: boolean
  onClose: () => void
}

const SHORTCUT_GROUPS = [
  {
    title: 'Camera',
    shortcuts: [
      { keys: ['R'], description: 'Reset camera to default view' },
      { keys: ['I'], description: 'Toggle isometric view' },
      { keys: ['Home'], description: 'Top-down view' },
      { keys: ['End'], description: 'Front view' },
      { keys: ['\u2190', '\u2192'], description: 'Rotate horizontally' },
      { keys: ['\u2191', '\u2193'], description: 'Rotate vertically' },
      { keys: ['+'], description: 'Zoom in' },
      { keys: ['\u2013'], description: 'Zoom out' },
    ],
  },
  {
    title: 'Selection',
    shortcuts: [
      { keys: ['Click'], description: 'Select patient point' },
      { keys: ['Shift', 'Drag'], description: 'Lasso select region' },
    ],
  },
  {
    title: 'Display',
    shortcuts: [
      { keys: ['Esc'], description: 'Exit fullscreen / collapse panels' },
      { keys: ['?'], description: 'Toggle this help overlay' },
    ],
  },
]

export function KeyboardShortcutsOverlay({ open, onClose }: KeyboardShortcutsOverlayProps) {
  // Close on Escape
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' || e.key === '?') {
        e.preventDefault()
        onClose()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl shadow-2xl border border-gray-200 p-6 max-w-md w-full mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Keyboard className="w-4 h-4 text-brand" />
            <h2 className="text-sm font-semibold text-slate-900">Keyboard Shortcuts</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-gray-100 text-slate-400 hover:text-slate-600"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-4">
          {SHORTCUT_GROUPS.map((group) => (
            <div key={group.title}>
              <h3 className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-2">
                {group.title}
              </h3>
              <div className="space-y-1.5">
                {group.shortcuts.map((s) => (
                  <div key={s.description} className="flex items-center justify-between">
                    <span className="text-xs text-slate-600">{s.description}</span>
                    <div className="flex items-center gap-1">
                      {s.keys.map((k, i) => (
                        <span key={i}>
                          <kbd className="px-1.5 py-0.5 text-[10px] font-mono font-medium bg-gray-100 text-slate-700 rounded border border-gray-200 shadow-sm">
                            {k}
                          </kbd>
                          {i < s.keys.length - 1 && (
                            <span className="text-[10px] text-slate-400 mx-0.5">+</span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 pt-3 border-t border-gray-100 text-center">
          <span className="text-[10px] text-slate-400">
            Press <kbd className="px-1 py-0.5 text-[10px] font-mono bg-gray-100 rounded border border-gray-200">?</kbd> to toggle
          </span>
        </div>
      </div>
    </div>
  )
}
