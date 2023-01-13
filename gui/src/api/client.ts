import { doesExist, NotImplementedError } from '@apextoaster/js-utils';

import { ConfigParams } from '../config.js';

export interface BaseImgParams {
  /**
   * Which ONNX model to use.
   */
  model?: string;

  /**
   * Hardware accelerator or CPU mode.
   */
  platform?: string;

  /**
   * Scheduling algorithm.
   */
  scheduler?: string;

  prompt: string;
  negativePrompt?: string;

  cfg: number;
  steps: number;
  seed: number;
}

export interface Img2ImgParams extends BaseImgParams {
  source: File;
  strength: number;
}

export type Img2ImgResponse = Required<Omit<Img2ImgParams, 'file'>>;

export interface Txt2ImgParams extends BaseImgParams {
  width?: number;
  height?: number;
}

export type Txt2ImgResponse = Required<Txt2ImgParams>;

export interface InpaintParams extends BaseImgParams {
  mask: Blob;
  source: File;
}

export interface OutpaintParams extends Img2ImgParams {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
}

export interface ApiResponse {
  output: {
    key: string;
    url: string;
  };
  params: Txt2ImgResponse;
}

export interface ApiReady {
  ready: boolean;
}

export interface ApiClient {
  models(): Promise<Array<string>>;
  params(): Promise<ConfigParams>;
  platforms(): Promise<Array<string>>;
  schedulers(): Promise<Array<string>>;

  img2img(params: Img2ImgParams): Promise<ApiResponse>;
  txt2img(params: Txt2ImgParams): Promise<ApiResponse>;

  inpaint(params: InpaintParams): Promise<ApiResponse>;
  outpaint(params: OutpaintParams): Promise<ApiResponse>;

  ready(params: ApiResponse): Promise<ApiReady>;
}

export const STATUS_SUCCESS = 200;

export function paramsFromConfig(defaults: ConfigParams): Required<BaseImgParams> {
  return {
    cfg: defaults.cfg.default,
    model: defaults.model.default,
    negativePrompt: defaults.negativePrompt.default,
    platform: defaults.platform.default,
    prompt: defaults.prompt.default,
    scheduler: defaults.scheduler.default,
    steps: defaults.steps.default,
    seed: defaults.seed.default,
  };
}

export function equalResponse(a: ApiResponse, b: ApiResponse): boolean {
  return a.output === b.output;
}

export function joinPath(...parts: Array<string>): string {
  return parts.join('/');
}

export function makeApiUrl(root: string, ...path: Array<string>) {
  return new URL(joinPath('api', ...path), root);
}

export function makeImageURL(root: string, type: string, params: BaseImgParams): URL {
  const url = makeApiUrl(root, type);
  url.searchParams.append('cfg', params.cfg.toFixed(1));
  url.searchParams.append('steps', params.steps.toFixed(0));

  if (doesExist(params.model)) {
    url.searchParams.append('model', params.model);
  }

  if (doesExist(params.platform)) {
    url.searchParams.append('platform', params.platform);
  }

  if (doesExist(params.scheduler)) {
    url.searchParams.append('scheduler', params.scheduler);
  }

  if (doesExist(params.seed)) {
    url.searchParams.append('seed', params.seed.toFixed(0));
  }

  // put prompt last, in case a load balancer decides to truncate the URL
  url.searchParams.append('prompt', params.prompt);

  if (doesExist(params.negativePrompt)) {
    url.searchParams.append('negativePrompt', params.negativePrompt);
  }

  return url;
}

export function makeClient(root: string, f = fetch): ApiClient {
  let pending: Promise<ApiResponse> | undefined;

  function throttleRequest(url: URL, options: RequestInit): Promise<ApiResponse> {
    return f(url, options).then((res) => parseApiResponse(root, res)).finally(() => {
      pending = undefined;
    });
  }

  return {
    async models(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'models');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async params(): Promise<ConfigParams> {
      const path = makeApiUrl(root, 'settings', 'params');
      const res = await f(path);
      return await res.json() as ConfigParams;
    },
    async schedulers(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'schedulers');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async platforms(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'platforms');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async img2img(params: Img2ImgParams): Promise<ApiResponse> {
      if (doesExist(pending)) {
        return pending;
      }

      const url = makeImageURL(root, 'img2img', params);

      const body = new FormData();
      body.append('source', params.source, 'source');

      pending = throttleRequest(url, {
        body,
        method: 'POST',
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
    async txt2img(params: Txt2ImgParams): Promise<ApiResponse> {
      if (doesExist(pending)) {
        return pending;
      }

      const url = makeImageURL(root, 'txt2img', params);

      if (doesExist(params.width)) {
        url.searchParams.append('width', params.width.toFixed(0));
      }

      if (doesExist(params.height)) {
        url.searchParams.append('height', params.height.toFixed(0));
      }

      pending = throttleRequest(url, {
        method: 'POST',
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
    async inpaint(params: InpaintParams) {
      if (doesExist(pending)) {
        return pending;
      }

      const url = makeImageURL(root, 'inpaint', params);

      const body = new FormData();
      body.append('mask', params.mask, 'mask');
      body.append('source', params.source, 'source');

      pending = throttleRequest(url, {
        body,
        method: 'POST',
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
    async outpaint() {
      throw new NotImplementedError();
    },
    async ready(params: ApiResponse): Promise<ApiReady> {
      const path = makeApiUrl(root, 'ready');
      path.searchParams.append('output', params.output.key);

      const res = await f(path);
      return await res.json() as ApiReady;
    }
  };
}

export async function parseApiResponse(root: string, res: Response): Promise<ApiResponse> {
  type LimitedResponse = Omit<ApiResponse, 'output'> & {output: string};

  if (res.status === STATUS_SUCCESS) {
    const data = await res.json() as LimitedResponse;
    const url = makeApiUrl(root, 'output', data.output).toString();
    return {
      output: {
        key: data.output,
        url,
      },
      params: data.params,
    };
  } else {
    throw new Error('request error');
  }
}
