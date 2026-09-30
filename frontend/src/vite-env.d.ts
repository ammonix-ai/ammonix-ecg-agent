/// <reference types="vite/client" />

/** App version injected at build time from `package.json` (Vite `define`). */
declare const __APP_VERSION__: string;

// Every VITE_* var the app consumes. All consumers use `|| fallback`, so
// optional typing is correct.
interface ImportMetaEnv {
  /**
   * Backend origin. Unset means same-origin, which the dev server proxies to
   * 127.0.0.1:8100. Read once, in api/client.ts, as `API_BASE`.
   */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
