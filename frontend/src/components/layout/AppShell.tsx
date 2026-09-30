import { Outlet } from 'react-router-dom';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { ScrollToTop } from '@/utils/ScrollToTop';
import { TopRibbon } from './TopRibbon';

/**
 * The whole chrome: a three-entry header and the page under it.
 *
 * Open build: the subtab ribbon, the breadcrumb strip and the persistent
 * background 3D canvas are gone. There are no subtabs to ribbon, the path is
 * one level deep, and every page rendered its own content over that canvas —
 * it was 63k points of hidden work on every route.
 */
export function AppShell() {
  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <ScrollToTop />
      <TopRibbon />
      <main className="flex-1 overflow-auto p-[clamp(1rem,0.5rem+1cqi,1.5rem)]">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
      <footer className="border-t border-[var(--border-default)] px-6 py-1.5 text-center text-xs text-text-muted">
        Research demonstration — not a medical device, not for clinical use.
      </footer>
    </div>
  );
}
