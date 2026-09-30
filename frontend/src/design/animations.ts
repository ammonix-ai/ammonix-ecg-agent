/**
 * Centralized Motion v12 animation variants.
 * Import from 'motion/react' — the v12 rebrand of Framer Motion.
 */
import type { Variants, Transition } from 'motion/react';

// --- Entrance animations ---

export const fadeIn: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
};

export const slideUp: Variants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: 12 },
};

export const slideDown: Variants = {
  initial: { opacity: 0, y: -12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -12 },
};

export const slideLeft: Variants = {
  initial: { opacity: 0, x: 20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: 20 },
};

export const scaleIn: Variants = {
  initial: { opacity: 0, scale: 0.95, y: 10 },
  animate: { opacity: 1, scale: 1, y: 0 },
  exit: { opacity: 0, scale: 0.95, y: 10 },
};

// --- Container variants ---

export const staggerContainer: Variants = {
  animate: {
    transition: {
      staggerChildren: 0.07,
    },
  },
};

export const staggerItem: Variants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
};

// --- Transitions ---

export const springGentle: Transition = {
  type: 'spring',
  duration: 0.4,
  bounce: 0,
};

export const springTab: Transition = {
  type: 'spring',
  stiffness: 500,
  damping: 30,
};

export const springBounce: Transition = {
  type: 'spring',
  stiffness: 400,
  damping: 25,
};

export const easeDefault: Transition = {
  duration: 0.35,
  ease: [0.25, 0.1, 0.25, 1],
};

export const easeFast: Transition = {
  duration: 0.15,
  ease: 'easeOut',
};

// --- Timing constants ---

export const duration = {
  fast: 0.15,
  normal: 0.2,
  slow: 0.35,
  stagger: 0.07,
  crossfade: 0.8,
} as const;

// --- Helpers ---

export function staggerDelay(index: number, base = duration.stagger): Transition {
  return { delay: index * base, ...easeDefault };
}
