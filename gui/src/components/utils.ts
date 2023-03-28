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
