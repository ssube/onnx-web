/* eslint-disable max-lines */
import { doesExist } from '@apextoaster/js-utils';

import { ServerParams } from './config.js';
import { range } from './utils.js';

/**
 * Shared parameters for anything using models, which is pretty much everything.
 */
export interface ModelParams {
  /**
   * The diffusion model to use.
   */
  model: string;

  /**
   * The hardware acceleration platform to use.
   */
  platform: string;

  /**
   * The upscaling model to use.
   */
  upscaling: string;

  /**
   * The correction model to use.
   */
  correction: string;

  /**
   * Use the long prompt weighting pipeline.
   */
  lpw: boolean;
}

/**
 * Shared parameters for most of the image requests.
 */
export interface BaseImgParams {
  scheduler: string;
  prompt: string;
  negativePrompt?: string;

  cfg: number;
  steps: number;
  seed: number;
}

/**
 * Parameters for txt2img requests.
 */
export interface Txt2ImgParams extends BaseImgParams {
  width?: number;
  height?: number;
}

/**
 * Parameters for img2img requests.
 */
export interface Img2ImgParams extends BaseImgParams {
  source: Blob;
  strength: number;
}

/**
 * Parameters for inpaint requests.
 */
export interface InpaintParams extends BaseImgParams {
  mask: Blob;
  source: Blob;

  filter: string;
  noise: string;
  strength: number;
  fillColor: string;
  tileOrder: string;
}

/**
 * Additional parameters for outpaint border.
 *
 * @todo should be nested under inpaint/outpaint params
 */
export interface OutpaintPixels {
  enabled: boolean;

  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * Parameters for outpaint requests.
 */
export type OutpaintParams = InpaintParams & OutpaintPixels;

/**
 * Additional parameters for the inpaint brush.
 *
 * These are not currently sent to the server and only stored in state.
 *
 * @todo move to state
 */
export interface BrushParams {
  color: number;
  size: number;
  strength: number;
}

/**
 * Additional parameters for upscaling.
 */
export interface UpscaleParams {
  enabled: boolean;
  upscaleOrder: string;

  denoise: number;
  scale: number;
  outscale: number;

  faces: boolean;
  faceStrength: number;
  faceOutscale: number;
}

/**
 * Parameters for upscale requests.
 */
export interface UpscaleReqParams {
  prompt: string;
  negativePrompt?: string;
  source: Blob;
}

export interface BlendParams {
  sources: Array<Blob>;
  mask: Blob;
}

/**
 * General response for most image requests.
 */
export interface ImageResponse {
  output: {
    key: string;
    url: string;
  };
  params: Required<BaseImgParams> & Required<ModelParams>;
  size: {
    width: number;
    height: number;
  };
}

/**
 * Status response from the ready endpoint.
 */
export interface ReadyResponse {
  progress: number;
  ready: boolean;
}

/**
 * List of available models.
 */
export interface ModelsResponse {
  diffusion: Array<string>;
  correction: Array<string>;
  upscaling: Array<string>;
}

export interface ApiClient {
  /**
   * List the available filter masks for inpaint.
   */
  masks(): Promise<Array<string>>;

  /**
   * List the available models.
   */
  models(): Promise<ModelsResponse>;

  /**
   * List the available noise sources for inpaint.
   */
  noises(): Promise<Array<string>>;

  /**
   * Get the valid server parameters to validate image parameters.
   */
  params(): Promise<ServerParams>;

  /**
   * Get the available hardware acceleration platforms.
   */
  platforms(): Promise<Array<string>>;

  /**
   * List the available pipeline schedulers.
   */
  schedulers(): Promise<Array<string>>;

  /**
   * Start a txt2img pipeline.
   */
  txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams): Promise<ImageResponse>;

  /**
   * Start an im2img pipeline.
   */
  img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams): Promise<ImageResponse>;

  /**
   * Start an inpaint pipeline.
   */
  inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams): Promise<ImageResponse>;

  /**
   * Start an outpaint pipeline.
   */
  outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams): Promise<ImageResponse>;

  /**
   * Start an upscale pipeline.
   */
  upscale(model: ModelParams, params: UpscaleReqParams, upscale?: UpscaleParams): Promise<ImageResponse>;

  /**
   * Start a blending pipeline.
   */
  blend(model: ModelParams, params: BlendParams, upscale?: UpscaleParams): Promise<ImageResponse>;

  /**
   * Check whether some pipeline's output is ready yet.
   */
  ready(params: ImageResponse): Promise<ReadyResponse>;

  cancel(params: ImageResponse): Promise<boolean>;
}

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

export function equalResponse(a: ImageResponse, b: ImageResponse): boolean {
  return a.output === b.output;
}

/**
 * Join URL path segments, which always use a forward slash per https://www.rfc-editor.org/rfc/rfc1738
 */
export function joinPath(...parts: Array<string>): string {
  return parts.join('/');
}

/**
 * Build the URL to an API endpoint, given the API root and a list of segments.
 */
export function makeApiUrl(root: string, ...path: Array<string>) {
  return new URL(joinPath('api', ...path), root);
}

/**
 * Build the URL for an image request, including all of the base image parameters.
 */
export function makeImageURL(root: string, type: string, params: BaseImgParams): URL {
  const url = makeApiUrl(root, type);
  url.searchParams.append('cfg', params.cfg.toFixed(FIXED_FLOAT));
  url.searchParams.append('steps', params.steps.toFixed(FIXED_INTEGER));

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

/**
 * Append the model parameters to an existing URL.
 */
export function appendModelToURL(url: URL, params: ModelParams) {
  url.searchParams.append('model', params.model);
  url.searchParams.append('platform', params.platform);
  url.searchParams.append('upscaling', params.upscaling);
  url.searchParams.append('correction', params.correction);
  url.searchParams.append('lpw', String(params.lpw));
}

/**
 * Append the upscale parameters to an existing URL.
 */
export function appendUpscaleToURL(url: URL, upscale: UpscaleParams) {
  url.searchParams.append('upscaleOrder', upscale.upscaleOrder);

  if (upscale.enabled) {
    url.searchParams.append('denoise', upscale.denoise.toFixed(FIXED_FLOAT));
    url.searchParams.append('scale', upscale.scale.toFixed(FIXED_INTEGER));
    url.searchParams.append('outscale', upscale.outscale.toFixed(FIXED_INTEGER));
  }

  if (upscale.faces) {
    url.searchParams.append('faces', String(upscale.faces));
    url.searchParams.append('faceOutscale', upscale.faceOutscale.toFixed(FIXED_INTEGER));
    url.searchParams.append('faceStrength', upscale.faceStrength.toFixed(FIXED_FLOAT));
  }
}

/**
 * Make an API client using the given API root and fetch client.
 */
export function makeClient(root: string, f = fetch): ApiClient {
  function throttleRequest(url: URL, options: RequestInit): Promise<ImageResponse> {
    return f(url, options).then((res) => parseApiResponse(root, res));
  }

  return {
    async masks(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'masks');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async models(): Promise<ModelsResponse> {
      const path = makeApiUrl(root, 'settings', 'models');
      const res = await f(path);
      return await res.json() as ModelsResponse;
    },
    async noises(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'noises');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async params(): Promise<ServerParams> {
      const path = makeApiUrl(root, 'settings', 'params');
      const res = await f(path);
      return await res.json() as ServerParams;
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
    async img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams): Promise<ImageResponse> {
      const url = makeImageURL(root, 'img2img', params);
      appendModelToURL(url, model);

      url.searchParams.append('strength', params.strength.toFixed(FIXED_FLOAT));

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      const body = new FormData();
      body.append('source', params.source, 'source');

      // eslint-disable-next-line no-return-await
      return await throttleRequest(url, {
        body,
        method: 'POST',
      });
    },
    async txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams): Promise<ImageResponse> {
      const url = makeImageURL(root, 'txt2img', params);
      appendModelToURL(url, model);

      if (doesExist(params.width)) {
        url.searchParams.append('width', params.width.toFixed(FIXED_INTEGER));
      }

      if (doesExist(params.height)) {
        url.searchParams.append('height', params.height.toFixed(FIXED_INTEGER));
      }

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      // eslint-disable-next-line no-return-await
      return await throttleRequest(url, {
        method: 'POST',
      });
    },
    async inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams) {
      const url = makeImageURL(root, 'inpaint', params);
      appendModelToURL(url, model);

      url.searchParams.append('filter', params.filter);
      url.searchParams.append('noise', params.noise);
      url.searchParams.append('strength', params.strength.toFixed(FIXED_FLOAT));
      url.searchParams.append('fillColor', params.fillColor);

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      const body = new FormData();
      body.append('mask', params.mask, 'mask');
      body.append('source', params.source, 'source');

      // eslint-disable-next-line no-return-await
      return await throttleRequest(url, {
        body,
        method: 'POST',
      });
    },
    async outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams) {
      const url = makeImageURL(root, 'inpaint', params);
      appendModelToURL(url, model);

      url.searchParams.append('filter', params.filter);
      url.searchParams.append('noise', params.noise);
      url.searchParams.append('strength', params.strength.toFixed(FIXED_FLOAT));
      url.searchParams.append('fillColor', params.fillColor);
      url.searchParams.append('tileOrder', params.tileOrder);

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

      // eslint-disable-next-line no-return-await
      return await throttleRequest(url, {
        body,
        method: 'POST',
      });
    },
    async upscale(model: ModelParams, params: UpscaleReqParams, upscale: UpscaleParams): Promise<ImageResponse> {
      const url = makeApiUrl(root, 'upscale');
      appendModelToURL(url, model);

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      url.searchParams.append('prompt', params.prompt);

      if (doesExist(params.negativePrompt)) {
        url.searchParams.append('negativePrompt', params.negativePrompt);
      }

      const body = new FormData();
      body.append('source', params.source, 'source');

      // eslint-disable-next-line no-return-await
      return await throttleRequest(url, {
        body,
        method: 'POST',
      });
    },
    async blend(model: ModelParams, params: BlendParams, upscale: UpscaleParams): Promise<ImageResponse> {
      const url = makeApiUrl(root, 'blend');
      appendModelToURL(url, model);

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      const body = new FormData();
      body.append('mask', params.mask, 'mask');

      for (const i of range(params.sources.length)) {
        const name = `source:${i.toFixed(0)}`;
        body.append(name, params.sources[i], name);
      }

      // eslint-disable-next-line no-return-await
      return await throttleRequest(url, {
        body,
        method: 'POST',
      });
    },
    async ready(params: ImageResponse): Promise<ReadyResponse> {
      const path = makeApiUrl(root, 'ready');
      path.searchParams.append('output', params.output.key);

      const res = await f(path);
      return await res.json() as ReadyResponse;
    },
    async cancel(params: ImageResponse): Promise<boolean> {
      const path = makeApiUrl(root, 'cancel');
      path.searchParams.append('output', params.output.key);

      const res = await f(path, {
        method: 'PUT',
      });
      return res.status === STATUS_SUCCESS;
    },
  };
}

/**
 * Parse a successful API response into the full image response record.
 *
 * The server sends over the output key, and the client is in the best position to turn
 * that into a full URL, since it already knows the root URL of the server.
 */
export async function parseApiResponse(root: string, res: Response): Promise<ImageResponse> {
  type LimitedResponse = Omit<ImageResponse, 'output'> & { output: string };

  if (res.status === STATUS_SUCCESS) {
    const data = await res.json() as LimitedResponse;
    const url = new URL(joinPath('output', data.output), root).toString();
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
