/* eslint-disable max-params */
/* eslint-disable camelcase */
/* eslint-disable max-lines */
import { doesExist, InvalidArgumentError, Maybe } from '@apextoaster/js-utils';
import { create as batcher, keyResolver, windowedFiniteBatchScheduler } from '@yornaath/batshit';

import { ServerParams } from '../config.js';
import { FIXED_FLOAT, FIXED_INTEGER, STATUS_SUCCESS } from '../constants.js';
import { JobResponse, JobResponseWithRetry, SuccessJobResponse } from '../types/api-v2.js';
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
  ExperimentalParams,
  HighresParams,
  ImageSize,
  Img2ImgJSONParams,
  Img2ImgParams,
  InpaintJSONParams,
  InpaintParams,
  ModelParams,
  OutpaintParams,
  OutpaintPixels,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
} from '../types/params.js';
import { range } from '../utils.js';
import { ApiClient } from './base.js';

export function equalResponse(a: JobResponse, b: JobResponse): boolean {
  return a.name === b.name;
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
export function makeApiURL(root: string, ...path: Array<string>): URL {
  return new URL(joinPath('api', ...path), root);
}

export interface ImageJSON {
  model: ModelParams;
  base: BaseImgParams;
  size?: ImageSize;
  border?: OutpaintPixels;
  upscale?: UpscaleParams;
  highres?: HighresParams;
  img2img?: Img2ImgJSONParams;
  inpaint?: InpaintJSONParams;
  experimental?: ExperimentalParams;
}

export interface JSONInner {
  [key: string]: string | number | boolean | undefined | JSONInner;
}

export interface JSONBody extends JSONInner {
  params: JSONInner;
}

export function makeImageJSON(params: ImageJSON): string {
  const { model, base, img2img, inpaint, size, border, upscale, highres, experimental } = params;

  const body: JSONBody = {
    device: {
      platform: model.platform,
    },
    params: {
      // model params
      model: model.model,
      pipeline: model.pipeline,
      upscaling: model.upscaling,
      correction: model.correction,
      control: model.control,
      // image params
      batch: base.batch,
      cfg: base.cfg,
      eta: base.eta,
      steps: base.steps,
      tiled_vae: base.tiled_vae,
      unet_overlap: base.unet_overlap,
      unet_tile: base.unet_tile,
      vae_overlap: base.vae_overlap,
      vae_tile: base.vae_tile,
      scheduler: base.scheduler,
      seed: base.seed,
      prompt: base.prompt,
      negativePrompt: base.negativePrompt,
    },
  };

  if (doesExist(img2img)) {
    body.params = {
      ...body.params,
      loopback: img2img.loopback,
      sourceFilter: img2img.sourceFilter,
      strength: img2img.strength,
    };
  }

  if (doesExist(inpaint)) {
    body.params = {
      ...body.params,
      filter: inpaint.filter,
      noise: inpaint.noise,
      strength: inpaint.strength,
      fillColor: inpaint.fillColor,
      tileOrder: inpaint.tileOrder,
    };
  }

  if (doesExist(size)) {
    body.size = {
      width: size.width,
      height: size.height,
    };
  }

  if (doesExist(border) && border.enabled) {
    body.border = {
      left: border.left,
      right: border.right,
      top: border.top,
      bottom: border.bottom,
    };
  }

  if (doesExist(upscale)) {
    body.upscale = {
      enabled: upscale.enabled,
      upscaleOrder: upscale.upscaleOrder,
      denoise: upscale.denoise,
      scale: upscale.scale,
      outscale: upscale.outscale,
      faces: upscale.faces,
      faceOutscale: upscale.faceOutscale,
      faceStrength: upscale.faceStrength,
    };
  }

  if (doesExist(highres)) {
    body.highres = {
      highres: highres.enabled,
      highresIterations: highres.highresIterations,
      highresMethod: highres.highresMethod,
      highresScale: highres.highresScale,
      highresSteps: highres.highresSteps,
      highresStrength: highres.highresStrength,
    };
  }

  if (doesExist(experimental)) {
    body.experimental = {
      latentSymmetry: {
        enabled: experimental.latentSymmetry.enabled,
        gradientStart: experimental.latentSymmetry.gradientStart,
        gradientEnd: experimental.latentSymmetry.gradientEnd,
        lineOfSymmetry: experimental.latentSymmetry.lineOfSymmetry,
      },
      promptEditing: {
        enabled: experimental.promptEditing.enabled,
        promptFilter: experimental.promptEditing.filter,
        removeTokens: experimental.promptEditing.removeTokens,
        addSuffix: experimental.promptEditing.addSuffix,
      },
    };
  }

  return JSON.stringify(body);
}

/**
 * Build the URL for an image request, including all of the base image parameters.
 *
 * @deprecated use `makeImageJSON` and `makeApiURL` instead
 */
export function makeImageURL(root: string, type: string, params: BaseImgParams): URL {
  const url = makeApiURL(root, type);
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
 *
 * @deprecated use `makeImageJSON` instead
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
 *
 * @deprecated use `makeImageJSON` instead
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

/**
 * Append the highres parameters to an existing URL.
 *
 * @deprecated use `makeImageJSON` instead
 */
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
export function makeClient(root: string, batchInterval: number, token: Maybe<string> = undefined, f = fetch): ApiClient {
  function parseRequest(url: URL, options: RequestInit): Promise<JobResponse> {
    return f(url, options).then((res) => parseJobResponse(root, res));
  }

  const client = {
    async extras(): Promise<ExtrasFile> {
      const path = makeApiURL(root, 'extras');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path);
      return await res.json() as ExtrasFile;
    },
    async writeExtras(extras: ExtrasFile): Promise<WriteExtrasResponse> {
      const path = makeApiURL(root, 'extras');

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
      const path = makeApiURL(root, 'settings', 'filters');
      const res = await f(path);
      return await res.json() as FilterResponse;
    },
    async models(): Promise<ModelResponse> {
      const path = makeApiURL(root, 'settings', 'models');
      const res = await f(path);
      return await res.json() as ModelResponse;
    },
    async noises(): Promise<Array<string>> {
      const path = makeApiURL(root, 'settings', 'noises');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async params(): Promise<ServerParams> {
      const path = makeApiURL(root, 'settings', 'params');
      const res = await f(path);
      return await res.json() as ServerParams;
    },
    async schedulers(): Promise<Array<string>> {
      const path = makeApiURL(root, 'settings', 'schedulers');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async pipelines(): Promise<Array<string>> {
      const path = makeApiURL(root, 'settings', 'pipelines');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async platforms(): Promise<Array<string>> {
      const path = makeApiURL(root, 'settings', 'platforms');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async strings(): Promise<Record<string, {
      translation: Record<string, string>;
    }>> {
      const path = makeApiURL(root, 'settings', 'strings');
      const res = await f(path);
      return await res.json() as Record<string, {
        translation: Record<string, string>;
      }>;
    },
    async wildcards(): Promise<Array<string>> {
      const path = makeApiURL(root, 'settings', 'wildcards');
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry> {
      const url = makeApiURL(root, 'img2img');
      const json = makeImageJSON({
        model,
        base: params,
        upscale,
        highres,
        img2img: params,
        experimental,
      });

      const form = new FormData();
      form.append('json', json);
      form.append('source', params.source, 'source');

      const job = await parseRequest(url, {
        body: form,
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
    async txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry> {
      const url = makeApiURL(root, 'txt2img');
      const json = makeImageJSON({
        model,
        base: params,
        size: params,
        upscale,
        highres,
        experimental,
      });

      const form = new FormData();
      form.append('json', json);

      const job = await parseRequest(url, {
        body: form,
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
    async inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry> {
      const url = makeApiURL(root, 'inpaint');
      const json = makeImageJSON({
        model,
        base: params,
        upscale,
        highres,
        inpaint: params,
        experimental,
      });

      const form = new FormData();
      form.append('json', json);
      form.append('mask', params.mask, 'mask');
      form.append('source', params.source, 'source');

      const job = await parseRequest(url, {
        body: form,
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
    async outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry> {
      const url = makeApiURL(root, 'inpaint');
      const json = makeImageJSON({
        model,
        base: params,
        upscale,
        highres,
        inpaint: params,
        experimental,
      });

      const form = new FormData();
      form.append('json', json);
      form.append('mask', params.mask, 'mask');
      form.append('source', params.source, 'source');

      const job = await parseRequest(url, {
        body: form,
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
      const url = makeApiURL(root, 'upscale');
      const json = makeImageJSON({
        model,
        base: params,
        upscale,
        highres,
      });

      const form = new FormData();
      form.append('json', json);
      form.append('source', params.source, 'source');

      const job = await parseRequest(url, {
        body: form,
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
      const url = makeApiURL(root, 'blend');
      const json = makeImageJSON({
        model,
        base: params as unknown as BaseImgParams, // TODO: fix this
        upscale,
      });

      const form = new FormData();
      form.append('json', json);
      form.append('mask', params.mask, 'mask');

      for (const i of range(params.sources.length)) {
        const name = `source:${i.toFixed(0)}`;
        form.append(name, params.sources[i], name);
      }

      const job = await parseRequest(url, {
        body: form,
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
      const url = makeApiURL(root, 'job');
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
      const path = makeApiURL(root, 'job', 'status');
      path.searchParams.append('jobs', keys.join(','));

      const res = await f(path);
      return await res.json() as Array<JobResponse>;
    },
    async cancel(keys: Array<string>): Promise<Array<JobResponse>> {
      const path = makeApiURL(root, 'job', 'cancel');
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
      const path = makeApiURL(root, 'worker', 'restart');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path, {
        method: 'POST',
      });
      return res.status === STATUS_SUCCESS;
    },
    async workers(): Promise<Array<unknown>> {
      const path = makeApiURL(root, 'worker', 'status');

      if (doesExist(token)) {
        path.searchParams.append('token', token);
      }

      const res = await f(path);
      return res.json();
    },
    outputURL(image: SuccessJobResponse, index: number): string {
      return new URL(joinPath('output', image.outputs[index]), root).toString();
    },
    thumbnailURL(image: SuccessJobResponse, index: number): Maybe<string> {
      if (doesExist(image.thumbnails) && doesExist(image.thumbnails[index])) {
        return new URL(joinPath('output', image.thumbnails[index]), root).toString();
      }

      return undefined;
    },
  };

  const batchStatus = batcher({
    fetcher: async (jobs: Array<string>) => client.status(jobs),
    resolver: keyResolver('name'),
    scheduler: windowedFiniteBatchScheduler({
      windowMs: batchInterval,
      maxBatchSize: 10,
    }),
  });

  return {
    ...client,
    async status(keys): Promise<Array<JobResponse>> {
      return Promise.all(keys.map((key) => batchStatus.fetch(key)));
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
