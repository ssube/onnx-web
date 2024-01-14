import { Breakpoint } from '@mui/material/styles';

export const BLEND_SOURCES = 2;

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
  limit: 8,

  /**
   * The number of additional images to be kept in history, so they can scroll
   * back into view when you delete one. Does not include deleted images.
   */
  scrollback: 4,
};

export const STANDARD_MARGIN = 4; // translated into 32px by mui
export const STANDARD_SPACING = 2;

export const LAYOUT_MIN = 300;
// eslint-disable-next-line @typescript-eslint/no-magic-numbers
export const LAYOUT_PROPORTIONS = [100, 200];

export const LAYOUT_STYLES = {
  horizontal: {
    container: false,
    control: {
      width: '30%',
    },
    direction: 'row',
    divider: 'vertical',
    history: {
      style: {
        ml: STANDARD_MARGIN,
      },
      width: 4,
    },
  },
  vertical: {
    container: 'lg' as Breakpoint,
    control: {
      width: undefined,
    },
    direction: 'column',
    divider: 'horizontal',
    history: {
      style: {
        mx: STANDARD_MARGIN,
        my: STANDARD_MARGIN,
      },
      width: 2,
    },
  },
} as const;

export const INITIAL_LOAD_TIMEOUT = 5_000;

export const STALE_TIME = 300_000; // 5 minutes
export const SAVE_TIME = 5_000; // 5 seconds

export const IMAGE_FILTER = '.bmp, .jpg, .jpeg, .png';
export const PARAM_VERSION = '>=0.10.0';

/**
 * Fixed precision for integer parameters.
 */
export const FIXED_INTEGER = 0;

/**
 * Fixed precision for float parameters.
 *
 * The GUI limits the input steps based on the server parameters, but this does limit
 * the maximum precision that can be sent back to the server, and may have to be
 * increased in the future.
 */
export const FIXED_FLOAT = 2;
export const STATUS_SUCCESS = 200;
