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

export type KeyFilter<T extends object> = {
  [K in keyof T]: T[K] extends number ? K : T[K] extends string ? K : never;
}[keyof T];

export type ConfigRanges<T extends object> = {
  [K in KeyFilter<T>]: T[K] extends number ? ConfigNumber : T[K] extends string ? ConfigString : never;
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
export const STALE_TIME = 3_000;

export async function loadConfig(): Promise<Config> {
  const configPath = new URL('./config.json', window.origin);
  const configReq = await fetch(configPath);
  if (configReq.status === STATUS_SUCCESS) {
    return configReq.json();
  } else {
    throw new Error('could not load config');
  }
}
