/* eslint-disable max-lines */
import { doesExist, InvalidArgumentError, Maybe } from '@apextoaster/js-utils';

import { ServerParams } from '../config.js';
import { range } from '../utils.js';
import {
  ApiClient,
  BaseImgParams,
  BlendParams,
  ChainPipeline,
  FilterResponse,
  HighresParams,
  ImageResponse,
  ImageResponseWithRetry,
  Img2ImgParams,
  InpaintParams,
  ModelParams,
  ModelResponse,
  OutpaintParams,
  ReadyResponse,
  RetryParams,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
  WriteExtrasResponse,
} from './types.js';
import { ExtrasFile } from '../types.js';

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
  url.searchParams.append('tiledVAE', String(params.tiledVAE));
  url.searchParams.append('tiles', params.tiles.toFixed(FIXED_INTEGER));
  url.searchParams.append('overlap', params.overlap.toFixed(FIXED_FLOAT));
  url.searchParams.append('stride', params.stride.toFixed(FIXED_INTEGER));

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
  url.searchParams.append('pipeline', params.pipeline);
  url.searchParams.append('platform', params.platform);
  url.searchParams.append('upscaling', params.upscaling);
  url.searchParams.append('correction', params.correction);
  url.searchParams.append('control', params.control);
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

export function appendHighresToURL(url: URL, highres: HighresParams) {
  if (highres.enabled) {
    url.searchParams.append('highres', String(highres.enabled));
    url.searchParams.append('highresIterations', highres.highresIterations.toFixed(FIXED_INTEGER));
    url.searchParams.append('highresMethod', highres.highresMethod);
    url.searchParams.append('highresScale', highres.highresScale.toFixed(FIXED_INTEGER));
    url.searchParams.append('highresSteps', highres.highresSteps.toFixed(FIXED_INTEGER));
    url.searchParams.append('highresStrength', highres.highresStrength.toFixed(FIXED_FLOAT));
  }
}

/**
 * Make an API client using the given API root and fetch client.
 */
export function makeClient(root: string, token: Maybe<string> = undefined, f = fetch): ApiClient {
  function parseRequest(url: URL, options: RequestInit): Promise<ImageResponse> {
    return f(url, options).then((res) => parseApiResponse(root, res));
  }

  return {
    async extras(): Promise<ExtrasFile> {
      const path = makeApiUrl(root, 'extras');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path);
      return await res.json() as ExtrasFile;
    },
    async writeExtras(extras: ExtrasFile): Promise<WriteExtrasResponse> {
      const path = makeApiUrl(root, 'extras');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path, {
        body: JSON.stringify(extras),
        method: 'PUT',
      });
      return await res.json() as WriteExtrasResponse;
    },
    async filters(): Promise<FilterResponse> {
      const path = makeApiUrl(root, 'settings', 'filters');
      const res = await f(path);
      return await res.json() as FilterResponse;
    },
    async models(): Promise<ModelResponse> {
      const path = makeApiUrl(root, 'settings', 'models');
      const res = await f(path);
      return await res.json() as ModelResponse;
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
    async pipelines(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'pipelines');
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
    async img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry> {
      const url = makeImageURL(root, 'img2img', params);
      appendModelToURL(url, model);

      url.searchParams.append('loopback', params.loopback.toFixed(FIXED_INTEGER));
      url.searchParams.append('strength', params.strength.toFixed(FIXED_FLOAT));

      if (doesExist(params.sourceFilter)) {
        url.searchParams.append('sourceFilter', params.sourceFilter);
      }

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      if (doesExist(highres)) {
        appendHighresToURL(url, highres);
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

      if (doesExist(highres)) {
        appendHighresToURL(url, highres);
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
    async inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry> {
      const url = makeImageURL(root, 'inpaint', params);
      appendModelToURL(url, model);

      url.searchParams.append('filter', params.filter);
      url.searchParams.append('noise', params.noise);
      url.searchParams.append('strength', params.strength.toFixed(FIXED_FLOAT));
      url.searchParams.append('fillColor', params.fillColor);

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      if (doesExist(highres)) {
        appendHighresToURL(url, highres);
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
    async outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry> {
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

      if (doesExist(highres)) {
        appendHighresToURL(url, highres);
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
    async upscale(model: ModelParams, params: UpscaleReqParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry> {
      const url = makeApiUrl(root, 'upscale');
      appendModelToURL(url, model);

      if (doesExist(upscale)) {
        appendUpscaleToURL(url, upscale);
      }

      if (doesExist(highres)) {
        appendHighresToURL(url, highres);
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
    async chain(model: ModelParams, chain: ChainPipeline): Promise<ImageResponse> {
      const url = makeApiUrl(root, 'chain');
      const body = JSON.stringify({
        ...chain,
        platform: model.platform,
      });

      // eslint-disable-next-line no-return-await
      return await parseRequest(url, {
        body,
        headers: {
          'Content-Type': 'application/json',
        },
        method: 'POST',
      });
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
          return this.img2img(retry.model, retry.params, retry.upscale, retry.highres);
        case 'inpaint':
          return this.inpaint(retry.model, retry.params, retry.upscale, retry.highres);
        case 'outpaint':
          return this.outpaint(retry.model, retry.params, retry.upscale, retry.highres);
        case 'txt2img':
          return this.txt2img(retry.model, retry.params, retry.upscale, retry.highres);
        case 'upscale':
          return this.upscale(retry.model, retry.params, retry.upscale, retry.highres);
        default:
          throw new InvalidArgumentError('unknown request type');
      }
    },
    async restart(): Promise<boolean> {
      const path = makeApiUrl(root, 'restart');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path, {
        method: 'POST',
      });
      return res.status === STATUS_SUCCESS;
    },
    async status(): Promise<Array<unknown>> {
      const path = makeApiUrl(root, 'status');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path);
      return res.json();
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
