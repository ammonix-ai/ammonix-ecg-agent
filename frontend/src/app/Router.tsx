import React, { Suspense } from 'react';
import { createBrowserRouter, Link, Navigate } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { t } from '@/i18n/t';

// --- Lazy page imports: one per nav entry, and that is the whole app ---
const UniversePage = React.lazy(() => import('@/pages/UniversePage'));
const AnalyzePage = React.lazy(() => import('@/pages/AnalyzePage'));
const AgentPage = React.lazy(() => import('@/pages/AgentPage'));

function lazySuspense(Component: React.LazyExoticComponent<React.ComponentType>) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-full bg-surface-0">
          <span className="text-body text-text-muted">
            {t('loading.fallback', 'Loading…')}
          </span>
        </div>
      }
    >
      <Component />
    </Suspense>
  );
}

function NotFound() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      <p className="text-h2 text-text-primary">404</p>
      <p className="text-body text-text-secondary">
        Nothing here. The open build has three pages.
      </p>
      <nav className="mt-2 flex gap-4 text-body font-semibold text-brand">
        <Link to="/universe" className="hover:underline">
          Universe
        </Link>
        <Link to="/analyze" className="hover:underline">
          Analyze
        </Link>
        <Link to="/agent" className="hover:underline">
          ECG Agent
        </Link>
      </nav>
    </div>
  );
}

/**
 * Routes retired with the sections they belonged to. The private monorepo's
 * Cohort (`/x/*`), World Model (`/phi/*`) and Skills (`/s/*`) pages, the
 * four-tab `/l/*` agent section and the two-mode `/dev` `/prod` homes are not
 * part of the open build — they drove backend routers this repo does not ship.
 * Old bookmarks land on the nearest live page instead of a dead 404.
 */
const LEGACY_REDIRECTS: Record<string, string> = {
  '/lambda/lattice': '/universe',
  '/lambda/training': '/universe',
  '/lambda/classifier': '/universe',
  '/lambda/kg': '/universe',
  '/lambda/tribes': '/universe',
  '/lambda/vocabulary': '/universe',
  '/l/inference': '/agent',
  '/l/models': '/agent',
  '/l/history': '/agent',
  '/l/agents': '/agent',
  '/prod/upload': '/analyze',
  '/prod/record': '/analyze',
  '/prod/results': '/analyze',
  '/dev': '/universe',
  '/prod': '/universe',
};

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: '/', element: <Navigate to="/universe" replace /> },

      { path: '/universe', element: lazySuspense(UniversePage) },
      { path: '/analyze', element: lazySuspense(AnalyzePage) },
      { path: '/agent', element: lazySuspense(AgentPage) },

      ...Object.entries(LEGACY_REDIRECTS).map(([from, to]) => ({
        path: from,
        element: <Navigate to={to} replace />,
      })),

      { path: '*', element: <NotFound /> },
    ],
  },
]);
