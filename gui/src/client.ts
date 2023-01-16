import { doesExist } from '@apextoaster/js-utils';

import { ConfigParams } from './config.js';

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
  source: Blob;
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
  source: Blob;

  filter: string;
  noise: string;
}

export interface OutpaintPixels {
  enabled: boolean;

  left: number;
  right: number;
  top: number;
  bottom: number;
}

export type OutpaintParams = InpaintParams & OutpaintPixels;

export interface BrushParams {
  color: number;
  size: number;
  strength: number;
}

export interface UpscaleParams {
  enabled: boolean;

  denoise: number;
  faces: boolean;
  scale: number;
}

export interface ApiResponse {
  output: {
    key: string;
    url: string;
  };
  params: Required<BaseImgParams>;
  size: {
    width: number;
    height: number;
  };
}

export interface ApiReady {
  ready: boolean;
}

export interface ApiClient {
  masks(): Promise<Array<string>>;
  models(): Promise<Array<string>>;
  noises(): Promise<Array<string>>;
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

export const FIXED_INTEGER = 0;
export const FIXED_FLOAT = 2;

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
  url.searchParams.append('cfg', params.cfg.toFixed(FIXED_FLOAT));
  url.searchParams.append('steps', params.steps.toFixed(FIXED_INTEGER));

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
    url.searchParams.append('seed', params.seed.toFixed(FIXED_INTEGER));
  }

  // put prompt last, in case a load balancer decides to truncate the URL
  url.searchParams.append('prompt', params.prompt);

  if (doesExist(params.negativePrompt)) {
    url.searchParams.append('negativePrompt', params.negativePrompt);
  }

  return url;
}

export function appendUpscaleToURL(url: URL, upscale: UpscaleParams) {
  url.searchParams.append('denoise', upscale.denoise.toFixed(FIXED_FLOAT));
  url.searchParams.append('faces', String(upscale.faces));
  url.searchParams.append('scale', upscale.scale.toFixed(FIXED_INTEGER));
}

export function makeClient(root: string, f = fetch): ApiClient {
  let pending: Promise<ApiResponse> | undefined;

  function throttleRequest(url: URL, options: RequestInit): Promise<ApiResponse> {
    return f(url, options).then((res) => parseApiResponse(root, res)).finally(() => {
      pending = undefined;
    });
  }

  return {
    async masks(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'masks');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async models(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'models');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async noises(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'noises');
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
    async img2img(params: Img2ImgParams, upscale?: UpscaleParams): Promise<ApiResponse> {
      if (doesExist(pending)) {
        return pending;
      }

      const url = makeImageURL(root, 'img2img', params);
      url.searchParams.append('strength', params.strength.toFixed(FIXED_FLOAT));

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      const body = new FormData();
      body.append('source', params.source, 'source');

      pending = throttleRequest(url, {
        body,
        method: 'POST',
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
    async txt2img(params: Txt2ImgParams, upscale?: UpscaleParams): Promise<ApiResponse> {
      if (doesExist(pending)) {
        return pending;
      }

      const url = makeImageURL(root, 'txt2img', params);

      if (doesExist(params.width)) {
        url.searchParams.append('width', params.width.toFixed(FIXED_INTEGER));
      }

      if (doesExist(params.height)) {
        url.searchParams.append('height', params.height.toFixed(FIXED_INTEGER));
      }

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      pending = throttleRequest(url, {
        method: 'POST',
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
    async inpaint(params: InpaintParams, upscale?: UpscaleParams) {
      if (doesExist(pending)) {
        return pending;
      }

      const url = makeImageURL(root, 'inpaint', params);
      url.searchParams.append('filter', params.filter);
      url.searchParams.append('noise', params.noise);
      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

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
    async outpaint(params: OutpaintParams, upscale?: UpscaleParams) {
      if (doesExist(pending)) {
        return pending;
      }

      const url = makeImageURL(root, 'inpaint', params);
      url.searchParams.append('filter', params.filter);
      url.searchParams.append('noise', params.noise);

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      if (doesExist(params.left)) {
        url.searchParams.append('left', params.left.toFixed(FIXED_INTEGER));
      }

      if (doesExist(params.right)) {
        url.searchParams.append('right', params.right.toFixed(FIXED_INTEGER));
      }

      if (doesExist(params.top)) {
        url.searchParams.append('top', params.top.toFixed(FIXED_INTEGER));
      }

      if (doesExist(params.bottom)) {
        url.searchParams.append('bottom', params.bottom.toFixed(FIXED_INTEGER));
      }

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
    async ready(params: ApiResponse): Promise<ApiReady> {
      const path = makeApiUrl(root, 'ready');
      path.searchParams.append('output', params.output.key);

      const res = await f(path);
      return await res.json() as ApiReady;
    }
  };
}

export async function parseApiResponse(root: string, res: Response): Promise<ApiResponse> {
  type LimitedResponse = Omit<ApiResponse, 'output'> & { output: string };

  if (res.status === STATUS_SUCCESS) {
    const data = await res.json() as LimitedResponse;
    const url = makeApiUrl(root, 'output', data.output).toString();
    return {
      ...data,
      output: {
        key: data.output,
        url,
      },
    };
  } else {
    throw new Error('request error');
  }
}
