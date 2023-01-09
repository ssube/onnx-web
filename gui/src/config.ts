import { Img2ImgParams, STATUS_SUCCESS, Txt2ImgParams } from './api/client.js';

export interface Config {
  api: {
    root: string;
  };
  default: {
    model: string;
    platform: string;
    scheduler: string;
    prompt: string;
  };
}

export async function loadConfig(): Promise<Config> {
  const configPath = new URL('./config.json', window.origin);
  const configReq = await fetch(configPath);
  if (configReq.status === STATUS_SUCCESS) {
    return configReq.json();
  } else {
    throw new Error('could not load config');
  }
}

export interface ConfigRange {
  default: number;
  min: number;
  max: number;
  step: number;
}

export type KeyFilter<T extends object> = {
  [K in keyof T]: T[K] extends number ? K : T[K] extends string ? K : never;
}[keyof T];

export type ConfigRanges<T extends object> = {
  [K in KeyFilter<T>]: T[K] extends number ? ConfigRange : T[K] extends string ? string : never;
};

export const DEFAULT_BRUSH = 8;
export const IMAGE_FILTER = '.bmp, .jpg, .jpeg, .png';
export const IMAGE_STEP = 8;
export const IMAGE_MAX = 512;

export const CONFIG_DEFAULTS: ConfigRanges<Required<Img2ImgParams & Txt2ImgParams>> = {
  cfg: {
    default: 6,
    min: 1,
    max: 30,
    step: 0.1,
  },
  height: {
    default: IMAGE_MAX,
    min: IMAGE_STEP,
    max: IMAGE_MAX,
    step: IMAGE_STEP,
  },
  model: '',
  negativePrompt: '',
  platform: '',
  prompt: 'an astronaut eating a hamburger',
  scheduler: '',
  steps: {
    default: 25,
    min: 1,
    max: 200,
    step: 1,
  },
  seed: {
    default: -1,
    min: -1,
    // eslint-disable-next-line @typescript-eslint/no-magic-numbers
    max: (2 ** 32) - 1,
    step: 1,
  },
  strength: {
    default: 0.5,
    min: 0,
    max: 1,
    step: 0.01,
  },
  width: {
    default: IMAGE_MAX,
    min: IMAGE_STEP,
    max: IMAGE_MAX,
    step: IMAGE_STEP,
  },
};

export const STALE_TIME = 3_000;
