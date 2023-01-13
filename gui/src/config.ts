import { Maybe } from '@apextoaster/js-utils';

import { Img2ImgParams, STATUS_SUCCESS, Txt2ImgParams } from './api/client.js';

export interface ConfigNumber {
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface ConfigString {
  default: string;
  keys: Array<string>;
}

export type KeyFilter<T extends object, TValid = number | string> = {
  [K in keyof T]: T[K] extends TValid ? K : never;
}[keyof T];

export type ConfigFiles<T extends object> = {
  [K in KeyFilter<T, Blob | File>]: Maybe<T[K]>;
};

export type ConfigRanges<T extends object> = {
  [K in KeyFilter<T>]: T[K] extends number ? ConfigNumber : T[K] extends string ? ConfigString : never;
};

export type ConfigState<T extends object, TValid = number | string> = {
  [K in KeyFilter<T, TValid>]: T[K] extends TValid ? T[K] : never;
};

export type ConfigParams = ConfigRanges<Required<Img2ImgParams & Txt2ImgParams>>;

export interface Config {
  api: {
    root: string;
  };
  params: {
    model: ConfigString;
    platform: ConfigString;
    scheduler: ConfigString;
    prompt: ConfigString;
  };
}

export const DEFAULT_BRUSH = {
  color: 255,
  size: 8,
};
export const IMAGE_FILTER = '.bmp, .jpg, .jpeg, .png';
export const STALE_TIME = 300_000; // 5 minutes
export const POLL_TIME = 5_000; // 5 seconds
export const SAVE_TIME = 5_000; // 5 seconds

export async function loadConfig(): Promise<Config> {
  const configPath = new URL('./config.json', window.origin);
  const configReq = await fetch(configPath);
  if (configReq.status === STATUS_SUCCESS) {
    return configReq.json();
  } else {
    throw new Error('could not load config');
  }
}
