/**
 * UI-only domain types for the Ammonix ECG Agent frontend.
 *
 * This file is reserved for **UI-config types** (`CardId`, `SubTab`,
 * `CardConfig`) consumed by `config/cards.ts` and the top nav.
 *
 * Per-domain view shapes live in their own `types/<x>.ts` files
 * (`types/ecg.ts`, `types/projection.ts`, `types/analyze.ts`).
 *
 * Open build: the nav is exactly three sections. The private monorepo's
 * `x` (Cohort), `phi` (World Model) and `s` (Skills) card ids are gone
 * along with their pages — they drove ~30 backend routers this repo does
 * not ship. The two-mode (developer/production) card sets are gone too;
 * there is one mode.
 */

export type CardId = 'universe' | 'analyze' | 'agent';

export interface SubTab {
  id: string;
  label: string;
  path: string;
}

export interface CardConfig {
  id: CardId;
  symbol: string;
  name: string;
  subtabs: SubTab[];
  activity: string;
}
