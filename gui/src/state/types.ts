import { PaletteMode } from '@mui/material';
import { StateCreator } from 'zustand';
import { ConfigFiles, ConfigState } from '../config.js';

export const MISSING_INDEX = -1;

export type Theme = PaletteMode | ''; // tri-state, '' is unset

/**
 * Combine optional files and required ranges.
 */
export type TabState<TabParams> = ConfigFiles<Required<TabParams>> & ConfigState<Required<TabParams>>;

/**
 * Shorthand for state creator to reduce repeated arguments.
 */
export type Slice<TState, TValue> = StateCreator<TState, [], [], TValue>;

/**
 * Default parameters for the inpaint brush.
 *
 * Not provided by the server yet.
 */
export const DEFAULT_BRUSH = {
  color: 255,
  size: 8,
  strength: 0.5,
};

/**
 * Default parameters for the image history.
 *
 * Not provided by the server yet.
 */
export const DEFAULT_HISTORY = {
  /**
   * The number of images to be shown.
   */
  limit: 4,

  /**
   * The number of additional images to be kept in history, so they can scroll
   * back into view when you delete one. Does not include deleted images.
   */
  scrollback: 2,
};
