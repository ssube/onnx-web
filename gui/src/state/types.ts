import { PaletteMode } from '@mui/material';
import { ConfigFiles, ConfigState } from '../config.js';

export const MISSING_INDEX = -1;

export type Theme = PaletteMode | ''; // tri-state, '' is unset

/**
 * Combine optional files and required ranges.
 */
export type TabState<TabParams> = ConfigFiles<Required<TabParams>> & ConfigState<Required<TabParams>>;
