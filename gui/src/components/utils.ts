import { Maybe, doesExist } from '@apextoaster/js-utils';
import { PaletteMode } from '@mui/material';

import { Theme } from '../state/types.js';
import { trimHash } from '../utils.js';

export const TAB_LABELS = [
  'txt2img',
  'img2img',
  'inpaint',
  'upscale',
  'blend',
  'models',
  'settings',
] as const;

export function getTab(hash: string): string {
  const route = trimHash(hash);
  if (route.length > 0) {
    return route;
  }

  return TAB_LABELS[0];
}

export function getTheme(currentTheme: Theme, preferDark: boolean): PaletteMode {
  if (currentTheme === '') {
    if (preferDark) {
      return 'dark';
    }
    return 'light';
  }
  return currentTheme as PaletteMode;
}

// eslint-disable-next-line @typescript-eslint/no-magic-numbers
export function getBatchInterval(defaultInterval = 5000): number {
  const query = new URLSearchParams(window.location.search);
  const interval = query.get('interval');
  if (doesExist(interval)) {
    return parseInt(interval, 10);
  }
  return defaultInterval;
}

export function getToken(): Maybe<string> {
  const query = new URLSearchParams(window.location.search);
  return query.get('token');
}
