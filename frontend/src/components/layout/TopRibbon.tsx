import { useEffect } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { Brain, Cpu, Server } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { motion } from 'motion/react';
import { CARDS, getActiveCard } from '@/config/cards';
import { springTab } from '@/design/animations';
import { cn } from '@/design/cn';
import { t } from '@/i18n/t';
import { useStatusStore } from '@/stores/statusStore';

/** unknown = the backend did not report this component; it is not "offline". */
type ServiceState = 'online' | 'probing' | 'offline' | 'unknown';

const STATE_COLOR: Record<ServiceState, string> = {
  online: 'bg-success',
  probing: 'bg-warning',
  offline: 'bg-danger',
  unknown: 'bg-text-muted',
};

const STATE_WORD: Record<ServiceState, string> = {
  online: 'online',
  probing: 'checking',
  offline: 'offline',
  unknown: 'not reported by the backend',
};

export function TopRibbon() {
  const { pathname } = useLocation();
  const activeCard = getActiveCard(pathname);

  const probe = useStatusStore((s) => s.probe);
  const loading = useStatusStore((s) => s.loading);
  const checkedAt = useStatusStore((s) => s.checkedAt);
  const refresh = useStatusStore((s) => s.refresh);

  useEffect(() => {
    if (checkedAt === null) void refresh();
  }, [checkedAt, refresh]);

  function flagState(flag: boolean | undefined): ServiceState {
    if (loading && checkedAt === null) return 'probing';
    if (!probe?.reachable) return 'offline';
    if (flag === undefined) return 'unknown';
    return flag ? 'online' : 'offline';
  }

  const backendState: ServiceState =
    loading && checkedAt === null ? 'probing' : probe?.reachable ? 'online' : 'offline';

  const services: { icon: LucideIcon; label: string; state: ServiceState }[] = [
    { icon: Server, label: 'Backend', state: backendState },
    { icon: Cpu, label: 'Analysis', state: flagState(probe?.status?.classifier) },
    { icon: Brain, label: 'Chat', state: flagState(probe?.status?.llm) },
  ];

  return (
    <header
      className="min-h-16 border-b border-[var(--border-default)] bg-surface-0/95 backdrop-blur-sm"
      style={{ containerType: 'inline-size' }}
    >
      <div className="relative flex min-h-16 items-center justify-between gap-4 px-6 py-2">
        <Link to="/universe" className="shrink-0 transition-opacity duration-200 hover:opacity-80">
          <span className="text-2xl font-extrabold tracking-tight text-brand">
            {t('brand.name', 'Ammonix ECG Agent')}
          </span>
        </Link>

        {/* The whole nav: three entries. */}
        <nav className="absolute left-1/2 flex -translate-x-1/2 items-center gap-6 @[32rem]:gap-8">
          {CARDS.map((card) => {
            const isActive = activeCard?.id === card.id;
            return (
              <NavLink
                key={card.id}
                to={card.path}
                data-nav-tab={card.id}
                title={card.activity}
                className={cn(
                  'group relative flex flex-col items-center gap-0.5 px-3 py-1.5 transition-colors',
                  isActive && 'rounded-[var(--radius-btn)] bg-brand-50/50',
                )}
              >
                <card.icon
                  size={18}
                  className={cn(
                    'transition-colors',
                    isActive ? 'text-brand' : 'text-text-secondary group-hover:text-text-primary',
                  )}
                />
                <span
                  className={cn(
                    'text-xs font-extrabold leading-tight tracking-tight transition-colors',
                    isActive ? 'text-brand' : 'text-text-secondary group-hover:text-text-primary',
                  )}
                >
                  {card.name}
                </span>
                {isActive && (
                  <motion.span
                    layoutId="activeNavTab"
                    className="absolute bottom-0 left-0 right-0 h-[2px] rounded-full bg-brand"
                    transition={springTab}
                  />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Status — click to re-probe GET /api/status */}
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={loading}
          title="Re-check GET /api/status"
          className="flex shrink-0 items-center gap-3 rounded-[var(--radius-btn)] px-2 py-1 transition-colors hover:bg-surface-2 disabled:opacity-60"
        >
          {services.map((svc) => (
            <span
              key={svc.label}
              className="flex items-center gap-1"
              title={`${svc.label}: ${STATE_WORD[svc.state]}`}
            >
              <svc.icon className="h-3 w-3 text-text-muted" aria-hidden />
              <span className={cn('h-1.5 w-1.5 rounded-full', STATE_COLOR[svc.state])} />
            </span>
          ))}
        </button>
      </div>
    </header>
  );
}
