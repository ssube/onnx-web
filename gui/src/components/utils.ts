import { PaletteMode } from '@mui/material';

import { Theme } from '../state.js';
import { trimHash } from '../utils.js';

export const TAB_LABELS = [
  'txt2img',
  'img2img',
  'inpaint',
  'upscale',
  'blend',
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
