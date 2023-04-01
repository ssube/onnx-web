/* eslint-disable max-lines */
import { doesExist, InvalidArgumentError } from '@apextoaster/js-utils';

import { ServerParams } from '../config.js';
import { range } from '../utils.js';

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

  batch: number;
  cfg: number;
  steps: number;
  seed: number;
  eta: number;
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
 * Additional parameters for upscaling. May be sent with most other requests to run a post-pipeline.
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

/**
 * Parameters for blend requests.
 */
export interface BlendParams {
  sources: Array<Blob>;
  mask: Blob;
}

export interface HighresParams {
  enabled: boolean;

  highresMethod: string;
  highresScale: number;
  highresSteps: number;
  highresStrength: number;
}

/**
 * Output image data within the response.
 */
export interface ImageOutput {
  key: string;
  url: string;
}

/**
 * Output image size, after upscaling and outscale.
 */
export interface ImageSize {
  width: number;
  height: number;
}

/**
 * General response for most image requests.
 */
export interface ImageResponse {
  outputs: Array<ImageOutput>;
  params: Required<BaseImgParams> & Required<ModelParams>;
  size: ImageSize;
}

/**
 * Status response from the ready endpoint.
 */
export interface ReadyResponse {
  cancelled: boolean;
  failed: boolean;
  progress: number;
  ready: boolean;
}

export interface NetworkModel {
  name: string;
  type: 'inversion' | 'lora';
  // TODO: add token
  // TODO: add layer/token count
}

/**
 * List of available models.
 */
export interface ModelsResponse {
  correction: Array<string>;
  diffusion: Array<string>;
  networks: Array<NetworkModel>;
  upscaling: Array<string>;
}

export type RetryParams = {
  type: 'txt2img';
  model: ModelParams;
  params: Txt2ImgParams;
  upscale?: UpscaleParams;
  highres?: HighresParams;
} | {
  type: 'img2img';
  model: ModelParams;
  params: Img2ImgParams;
  upscale?: UpscaleParams;
} | {
  type: 'inpaint';
  model: ModelParams;
  params: InpaintParams;
  upscale?: UpscaleParams;
} | {
  type: 'outpaint';
  model: ModelParams;
  params: OutpaintParams;
  upscale?: UpscaleParams;
} | {
  type: 'upscale';
  model: ModelParams;
  params: UpscaleReqParams;
  upscale?: UpscaleParams;
} | {
  type: 'blend';
  model: ModelParams;
  params: BlendParams;
  upscale?: UpscaleParams;
};

export interface ImageResponseWithRetry {
  image: ImageResponse;
  retry: RetryParams;
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
   * Load extra strings from the server.
   */
  strings(): Promise<Record<string, {
    translation: Record<string, string>;
  }>>;

  /**
   * Start a txt2img pipeline.
   */
  txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an im2img pipeline.
   */
  img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an inpaint pipeline.
   */
  inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an outpaint pipeline.
   */
  outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an upscale pipeline.
   */
  upscale(model: ModelParams, params: UpscaleReqParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry>;

  /**
   * Start a blending pipeline.
   */
  blend(model: ModelParams, params: BlendParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry>;

  /**
   * Check whether some pipeline's output is ready yet.
   */
  ready(key: string): Promise<ReadyResponse>;

  cancel(key: string): Promise<boolean>;

  retry(params: RetryParams): Promise<ImageResponseWithRetry>;
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
  return a.outputs === b.outputs;
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
  url.searchParams.append('batch', params.batch.toFixed(FIXED_INTEGER));
  url.searchParams.append('cfg', params.cfg.toFixed(FIXED_FLOAT));
  url.searchParams.append('eta', params.eta.toFixed(FIXED_FLOAT));
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
  function parseRequest(url: URL, options: RequestInit): Promise<ImageResponse> {
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
    async strings(): Promise<Record<string, {
      translation: Record<string, string>;
    }>> {
      const path = makeApiUrl(root, 'settings', 'strings');
      const res = await f(path);
      return await res.json() as Record<string, {
        translation: Record<string, string>;
      }>;
    },
    async img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry> {
      const url = makeImageURL(root, 'img2img', params);
      appendModelToURL(url, model);

      url.searchParams.append('strength', params.strength.toFixed(FIXED_FLOAT));

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      const body = new FormData();
      body.append('source', params.source, 'source');

      const image = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        image,
        retry: {
          type: 'img2img',
          model,
          params,
          upscale,
        },
      };
    },
    async txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry> {
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

      if (doesExist(highres) && highres.enabled) {
        url.searchParams.append('highresMethod', highres.highresMethod);
        url.searchParams.append('highresScale', highres.highresScale.toFixed(FIXED_INTEGER));
        url.searchParams.append('highresSteps', highres.highresSteps.toFixed(FIXED_INTEGER));
        url.searchParams.append('highresStrength', highres.highresStrength.toFixed(FIXED_FLOAT));
      }

      const image = await parseRequest(url, {
        method: 'POST',
      });
      return {
        image,
        retry: {
          type: 'txt2img',
          model,
          params,
          upscale,
          highres,
        },
      };
    },
    async inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry> {
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

      const image = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        image,
        retry: {
          type: 'inpaint',
          model,
          params,
          upscale,
        },
      };
    },
    async outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry> {
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

      const image = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        image,
        retry: {
          type: 'outpaint',
          model,
          params,
          upscale,
        },
      };
    },
    async upscale(model: ModelParams, params: UpscaleReqParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry> {
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

      const image = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        image,
        retry: {
          type: 'upscale',
          model,
          params,
          upscale,
        },
      };
    },
    async blend(model: ModelParams, params: BlendParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry> {
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

      const image = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        image,
        retry: {
          type: 'blend',
          model,
          params,
          upscale,
        }
      };
    },
    async ready(key: string): Promise<ReadyResponse> {
      const path = makeApiUrl(root, 'ready');
      path.searchParams.append('output', key);

      const res = await f(path);
      return await res.json() as ReadyResponse;
    },
    async cancel(key: string): Promise<boolean> {
      const path = makeApiUrl(root, 'cancel');
      path.searchParams.append('output', key);

      const res = await f(path, {
        method: 'PUT',
      });
      return res.status === STATUS_SUCCESS;
    },
    async retry(retry: RetryParams): Promise<ImageResponseWithRetry> {
      switch (retry.type) {
        case 'blend':
          return this.blend(retry.model, retry.params, retry.upscale);
        case 'img2img':
          return this.img2img(retry.model, retry.params, retry.upscale);
        case 'inpaint':
          return this.inpaint(retry.model, retry.params, retry.upscale);
        case 'outpaint':
          return this.outpaint(retry.model, retry.params, retry.upscale);
        case 'txt2img':
          return this.txt2img(retry.model, retry.params, retry.upscale, retry.highres);
        case 'upscale':
          return this.upscale(retry.model, retry.params, retry.upscale);
        default:
          throw new InvalidArgumentError('unknown request type');
      }
    }
  };
}

/**
 * Parse a successful API response into the full image response record.
 *
 * The server sends over the output key, and the client is in the best position to turn
 * that into a full URL, since it already knows the root URL of the server.
 */
export async function parseApiResponse(root: string, res: Response): Promise<ImageResponse> {
  type LimitedResponse = Omit<ImageResponse, 'outputs'> & { outputs: Array<string> };

  if (res.status === STATUS_SUCCESS) {
    const data = await res.json() as LimitedResponse;

    const outputs = data.outputs.map((output) => {
      const url = new URL(joinPath('output', output), root).toString();
      return {
        key: output,
        url,
      };
    });

    return {
      ...data,
      outputs,
    };
  } else {
    throw new Error('request error');
  }
}
