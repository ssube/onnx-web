import { doesExist, Maybe } from '@apextoaster/js-utils';
import { merge } from 'lodash';

import { STATUS_SUCCESS } from './client/api.js';
import {
  HighresParams,
  Img2ImgParams,
  InpaintParams,
  ModelParams,
  OutpaintParams,
  Txt2ImgParams,
  UpscaleParams,
} from './client/types.js';

export interface ConfigBoolean {
  default: boolean;
}

export interface ConfigNumber {
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface ConfigString {
  default: string;
  keys: Record<string, string>;
}

/**
 * Helper type to filter keys whose value extends `TValid`.
 */
export type KeyFilter<T extends object, TValid = boolean | number | string> = {
  [K in keyof T]: T[K] extends TValid ? K : never;
}[keyof T];

/**
 * Keep fields with a file-like value, but make them optional.
 */
export type ConfigFiles<T extends object> = {
  [K in KeyFilter<T, Blob | File | Array<Blob | File>>]: Maybe<T[K]>;
};

/**
 * Map numbers and strings to their corresponding config types and drop the rest of the fields.
 */
export type ConfigRanges<T extends object> = {
  [K in KeyFilter<T>]: T[K] extends boolean ? ConfigBoolean : T[K] extends number ? ConfigNumber : T[K] extends string ? ConfigString : never;
};

/**
 * Keep fields whose value extends `TValid` and drop the rest.
 */
export type ConfigState<T extends object, TValid = boolean | number | string> = {
  [K in KeyFilter<T, TValid>]: T[K] extends TValid ? T[K] : never;
};

// eslint does not understand how to indent this and expects each line to increase
/* eslint-disable */
/**
 * Combine all of the request parameter groups, make optional parameters required, then
 * map them to the number/string ranges.
 */
export type ServerParams = ConfigRanges<Required<
  Img2ImgParams &
  Txt2ImgParams &
  InpaintParams &
  ModelParams &
  OutpaintParams &
  UpscaleParams &
  HighresParams
>> & {
  version: string;
};
/* eslint-enable */

/**
 * Parameters that can be customized on the client, through the config file or settings tab.
 */
export interface ClientParams {
  model: ConfigString;
  platform: ConfigString;
  scheduler: ConfigString;
  prompt: ConfigString;
}

export interface Config<T = ClientParams> {
  api: {
    root: string;
  };
  params: T;
}

export const IMAGE_FILTER = '.bmp, .jpg, .jpeg, .png';
export const PARAM_VERSION = '>=0.10.0';

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

export function mergeConfig(client: Config, server: ServerParams): Config<ServerParams> {
  const full = merge({}, server, client.params);;

  return {
    ...client,
    params: full,
  };
}

export function getApiRoot(config: Config): string {
  const query = new URLSearchParams(window.location.search);
  const api = query.get('api');

  if (doesExist(api)) {
    return api;
  } else {
    return config.api.root;
  }
}

export function isDebug(): boolean {
  const query = new URLSearchParams(window.location.search);
  const debug = query.get('debug');

  if (doesExist(debug)) {
    const val = debug.toLowerCase();
    // eslint-disable-next-line no-restricted-syntax
    return val === '1' || val === 't' || val === 'true' || val === 'y' || val === 'yes';
  } else {
    return false;
  }
}
