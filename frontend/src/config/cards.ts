import { Shell, Activity, MessagesSquare } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { CardId } from '@/types';

export interface CardDef {
  id: CardId;
  /** Route the nav entry links to. */
  path: string;
  name: string;
  icon: LucideIcon;
  /** One-line description of what the section does. */
  activity: string;
  accent: string;
}

/**
 * The whole nav. Three entries, no subtabs, no modes.
 *
 * Universe is the fixed 63,256-point embedding; Analyze runs the frozen
 * classifier over a shipped or uploaded recording; ECG Agent is the chat that
 * *verifies* what the classifier said against the trace.
 */
export const CARDS: CardDef[] = [
  {
    id: 'universe',
    path: '/universe',
    name: 'Universe',
    icon: Shell,
    activity: '3D diagnosis space',
    accent: '#2743B8',
  },
  {
    id: 'analyze',
    path: '/analyze',
    name: 'Analyze',
    icon: Activity,
    activity: 'Analyze a recording',
    accent: '#0E7490',
  },
  {
    id: 'agent',
    path: '/agent',
    name: 'ECG Agent',
    icon: MessagesSquare,
    activity: 'Verify the read',
    accent: '#B45309',
  },
];

export function getCard(id: CardId): CardDef | undefined {
  return CARDS.find((c) => c.id === id);
}

/** Which nav entry owns this pathname, if any. */
export function getActiveCard(pathname: string): CardDef | undefined {
  return CARDS.find((c) => pathname === c.path || pathname.startsWith(`${c.path}/`));
}
