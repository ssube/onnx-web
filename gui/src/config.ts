import { Maybe } from '@apextoaster/js-utils';

import { Img2ImgParams, InpaintParams, ModelParams, OutpaintParams, STATUS_SUCCESS, Txt2ImgParams, UpscaleParams } from './client.js';

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

/* eslint-disable */
export type ConfigParams = ConfigRanges<Required<
  Img2ImgParams &
  Txt2ImgParams &
  InpaintParams &
  ModelParams &
  OutpaintParams &
  UpscaleParams
>> & {
  version: string;
};
/* eslint-enable */

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
export const PARAM_VERSION = '>=0.4.0';

export const STALE_TIME = 300_000; // 5 minutes
export const POLL_TIME = 5_000; // 5 seconds
export const SAVE_TIME = 5_000; // 5 seconds

export async function loadConfig(): Promise<Config> {
  const configPath = new URL('./config.json', window.location.href);
  const configReq = await fetch(configPath);
  if (configReq.status === STATUS_SUCCESS) {
    return configReq.json();
  } else {
    throw new Error('could not load config');
  }
}
