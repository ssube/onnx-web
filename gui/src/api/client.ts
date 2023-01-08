import { doesExist } from '@apextoaster/js-utils';

export interface Img2ImgParams {
  model?: string;
  platform?: string;
  scheduler?: string;

  prompt: string;
  cfg: number;
  steps: number;

  seed?: number;

  source: File;
}

export interface Txt2ImgParams {
  model?: string;
  platform?: string;
  scheduler?: string;

  prompt: string;
  cfg: number;
  steps: number;

  width?: number;
  height?: number;
  seed?: number;
}

export interface Txt2ImgResponse extends Txt2ImgParams {
  model: string;
  platform: string;
  scheduler: string;

  width: number;
  height: number;
  seed: number;
}

export interface ApiResponse {
  output: string;
  params: Txt2ImgResponse;
}

export interface ApiClient {
  models(): Promise<Array<string>>;
  platforms(): Promise<Array<string>>;
  schedulers(): Promise<Array<string>>;

  img2img(params: Img2ImgParams): Promise<ApiResponse>; // TODO: slightly different response type
  txt2img(params: Txt2ImgParams): Promise<ApiResponse>;
}

export const STATUS_SUCCESS = 200;

export function joinPath(...parts: Array<string>): string {
  return parts.join('/');
}

export async function imageFromResponse(root: string, res: Response): Promise<ApiResponse> {
  if (res.status === STATUS_SUCCESS) {
    const data = await res.json() as ApiResponse;
    const output = new URL(joinPath('output', data.output), root).toString();
    return {
      output,
      params: data.params,
    };
  } else {
    throw new Error('request error');
  }
}

export function makeClient(root: string, f = fetch): ApiClient {
  let pending: Promise<ApiResponse> | undefined;

  return {
    async models(): Promise<Array<string>> {
      const path = new URL(joinPath('settings', 'models'), root);
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async schedulers(): Promise<Array<string>> {
      const path = new URL(joinPath('settings', 'schedulers'), root);
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async platforms(): Promise<Array<string>> {
      const path = new URL(joinPath('settings', 'platforms'), root);
      const res = await f(path);
      return await res.json() as Array<string>;
    },
    async img2img(params: Img2ImgParams): Promise<ApiResponse> {
      if (doesExist(pending)) {
        return pending;
      }

      const url = new URL('img2img', root);
      url.searchParams.append('cfg', params.cfg.toFixed(0));
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

      url.searchParams.append('prompt', params.prompt);

      const body = new FormData();
      body.append('source', params.source, 'source');

      pending = f(url, {
        body,
        method: 'POST',
      }).then((res) => imageFromResponse(root, res)).finally(() => {
        pending = undefined;
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
    async txt2img(params: Txt2ImgParams): Promise<ApiResponse> {
      if (doesExist(pending)) {
        return pending;
      }

      const url = new URL('txt2img', root);
      url.searchParams.append('cfg', params.cfg.toFixed(0));
      url.searchParams.append('steps', params.steps.toFixed(0));

      if (doesExist(params.width)) {
        url.searchParams.append('width', params.width.toFixed(0));
      }

      if (doesExist(params.height)) {
        url.searchParams.append('height', params.height.toFixed(0));
      }

      if (doesExist(params.seed)) {
        url.searchParams.append('seed', params.seed.toFixed(0));
      }

      if (doesExist(params.model)) {
        url.searchParams.append('model', params.model);
      }

      if (doesExist(params.platform)) {
        url.searchParams.append('platform', params.platform);
      }

      if (doesExist(params.scheduler)) {
        url.searchParams.append('scheduler', params.scheduler);
      }

      url.searchParams.append('prompt', params.prompt);

      pending = f(url, {
        method: 'POST',
      }).then((res) => imageFromResponse(root, res)).finally(() => {
        pending = undefined;
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
  };
}
