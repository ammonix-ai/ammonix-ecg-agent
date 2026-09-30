/**
 * String externalization stub.
 * Returns the fallback (English) string directly.
 * V3-MOCK: wire i18next or similar at merge for DE/FR/IT support.
 */
export function t(_key: string, fallback: string): string {
  return fallback;
}
