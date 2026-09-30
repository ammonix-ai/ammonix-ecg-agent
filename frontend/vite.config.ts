import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { resolve } from 'path';
import { readFileSync } from 'fs';

const pkg = JSON.parse(readFileSync(resolve(__dirname, 'package.json'), 'utf-8')) as { version: string };

// Ammonix ECG Agent backend: 127.0.0.1:8100 in dev. Override with VITE_API_URL
// (e.g. VITE_API_URL=http://127.0.0.1:9000 npm run dev) when the backend runs
// somewhere else. Keep this in sync with DEV_API_ORIGIN in src/api/client.ts.
//
// Requests leave the browser same-origin and are proxied here, so the backend
// needs no CORS grant for the dev setup and POST /api/analyze +
// POST /api/chat/stream are not subject to a preflight.
const API_TARGET = process.env.VITE_API_URL || 'http://127.0.0.1:8100';

export default defineConfig({
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  // V3.26 (Q-A3-21 option b): build-only strip of console.* + debugger.
  // esbuild config applies to `vite build`; `vite` (dev) and `vitest`
  // transforms keep them so dev DevTools still surface logs and tests
  // can assert on console output. Drop reduces prod bundle noise without
  // touching source code (the alternative — wrapping every call site in
  // `if (import.meta.env.DEV)` — was rejected as 251-site churn).
  esbuild: {
    drop: ['console', 'debugger'],
  },
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    // 5174, not 5173: 5173 belongs to the private app this repo was extracted
    // from, and two dev servers fighting over one port is a bad first run.
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        // POST /api/chat/stream is an SSE response; buffering it would make the
        // agent look frozen until the model finished.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform';
            }
          });
        },
      },
      '/health': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Manual chunk grouping for cache stability + bundle clarity. Only
        // the three.js stack is worth splitting out — it is ~1 MB and is
        // needed by exactly one of the three pages.
        //
        // The `three` chunk transitively ships zustand@5 because
        // @react-three/fiber@9 + drei@10 depend on it internally, while the
        // app pins zustand@4.5. npm gives each major its own nested install,
        // so module instances are isolated (no runtime conflict) but the
        // bundle carries both. App code MUST NOT share a store between
        // three.js components and the rest of the app — they would resolve to
        // different zustand instances and silently drift out of sync.
        manualChunks: {
          'three': ['three', '@react-three/fiber', '@react-three/drei'],
        },
      },
    },
  },
});
