/* eslint-disable max-lines */
import { doesExist, InvalidArgumentError, Maybe } from '@apextoaster/js-utils';

import { ServerParams } from '../config.js';
import {
  FilterResponse,
  ModelResponse,
  RetryParams,
  WriteExtrasResponse,
} from '../types/api.js';
import { ChainPipeline } from '../types/chain.js';
import { ExtrasFile } from '../types/model.js';
import {
  BaseImgParams,
  BlendParams,
  HighresParams,
  Img2ImgParams,
  InpaintParams,
  ModelParams,
  OutpaintParams,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
} from '../types/params.js';
import { range } from '../utils.js';
import { ApiClient } from './base.js';
import { JobResponse, JobResponseWithRetry, SuccessJobResponse } from '../types/api-v2.js';

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

export function equalResponse(a: JobResponse, b: JobResponse): boolean {
  return a.name === b.name && a.status === b.status && a.type === b.type;
  // return a.outputs === b.outputs;
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
  url.searchParams.append('tiled_vae', String(params.tiled_vae));
  url.searchParams.append('unet_overlap', params.unet_overlap.toFixed(FIXED_FLOAT));
  url.searchParams.append('unet_tile', params.unet_tile.toFixed(FIXED_INTEGER));
  url.searchParams.append('vae_overlap', params.vae_overlap.toFixed(FIXED_FLOAT));
  url.searchParams.append('vae_tile', params.vae_tile.toFixed(FIXED_INTEGER));

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
  url.searchParams.append('upscale', String(upscale.enabled));
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
  function parseRequest(url: URL, options: RequestInit): Promise<JobResponse> {
    return f(url, options).then((res) => parseJobResponse(root, res));
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
    async wildcards(): Promise<Array<string>> {
      const path = makeApiUrl(root, 'settings', 'wildcards');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<JobResponseWithRetry> {
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

      const job = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        job,
        retry: {
          type: 'img2img',
          model,
          params,
          upscale,
        },
      };
    },
    async txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<JobResponseWithRetry> {
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

      const job = await parseRequest(url, {
        method: 'POST',
      });
      return {
        job,
        retry: {
          type: 'txt2img',
          model,
          params,
          upscale,
          highres,
        },
      };
    },
    async inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<JobResponseWithRetry> {
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

      const job = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        job,
        retry: {
          type: 'inpaint',
          model,
          params,
          upscale,
        },
      };
    },
    async outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<JobResponseWithRetry> {
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

      const job = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        job,
        retry: {
          type: 'outpaint',
          model,
          params,
          upscale,
        },
      };
    },
    async upscale(model: ModelParams, params: UpscaleReqParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<JobResponseWithRetry> {
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

      const job = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        job,
        retry: {
          type: 'upscale',
          model,
          params,
          upscale,
        },
      };
    },
    async blend(model: ModelParams, params: BlendParams, upscale?: UpscaleParams): Promise<JobResponseWithRetry> {
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

      const job = await parseRequest(url, {
        body,
        method: 'POST',
      });
      return {
        job,
        retry: {
          type: 'blend',
          model,
          params,
          upscale,
        }
      };
    },
    async chain(model: ModelParams, chain: ChainPipeline): Promise<JobResponse> {
      const url = makeApiUrl(root, 'job');
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
    async status(keys: Array<string>): Promise<Array<JobResponse>> {
      const path = makeApiUrl(root, 'job', 'status');
      path.searchParams.append('jobs', keys.join(','));

      const res = await f(path);
      return await res.json() as Array<JobResponse>;
    },
    async cancel(keys: Array<string>): Promise<Array<JobResponse>> {
      const path = makeApiUrl(root, 'job', 'cancel');
      path.searchParams.append('jobs', keys.join(','));

      const res = await f(path, {
        method: 'PUT',
      });
      return await res.json() as Array<JobResponse>;
    },
    async retry(retry: RetryParams): Promise<JobResponseWithRetry> {
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
      const path = makeApiUrl(root, 'worker', 'restart');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path, {
        method: 'POST',
      });
      return res.status === STATUS_SUCCESS;
    },
    async workers(): Promise<Array<unknown>> {
      const path = makeApiUrl(root, 'worker', 'status');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path);
      return res.json();
    },
    outputURL(image: SuccessJobResponse, index: number): string {
      return new URL(joinPath('output', image.outputs[index]), root).toString();
    },
  };
}

/**
 * Parse a successful API response into the full image response record.
 *
 * The server sends over the output key, and the client is in the best position to turn
 * that into a full URL, since it already knows the root URL of the server.
 */
export async function parseJobResponse(root: string, res: Response): Promise<JobResponse> {
  if (res.status === STATUS_SUCCESS) {
    return await res.json() as JobResponse;
  } else {
    throw new Error('request error');
  }
}
