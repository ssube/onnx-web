import { doesExist } from '@apextoaster/js-utils';

export interface Txt2ImgParams {
  prompt: string;
  cfg: number;
  steps: number;
  width?: number;
  height?: number;
  seed?: string;
  scheduler?: string;
}

export interface ApiResponse {
  output: string;
  params: Txt2ImgParams;
}

export interface ApiClient {
  txt2img(params: Txt2ImgParams): Promise<ApiResponse>;
}

export const STATUS_SUCCESS = 200;

export async function imageFromResponse(root: string, res: Response): Promise<ApiResponse> {
  if (res.status === STATUS_SUCCESS) {
    const data = await res.json() as ApiResponse;
    const output = new URL(['output', data.output].join('/'), root).toString();
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
    async txt2img(params: Txt2ImgParams): Promise<ApiResponse> {
      if (doesExist(pending)) {
        return pending;
      }

      const url = new URL('/txt2img', root);
      url.searchParams.append('cfg', params.cfg.toFixed(0));
      url.searchParams.append('steps', params.steps.toFixed(0));

      if (doesExist(params.width)) {
        url.searchParams.append('width', params.width.toFixed(0));
      }

      if (doesExist(params.height)) {
        url.searchParams.append('height', params.height.toFixed(0));
      }

      if (doesExist(params.seed)) {
        url.searchParams.append('seed', params.seed);
      }

      if (doesExist(params.scheduler)) {
        url.searchParams.append('scheduler', params.scheduler);
      }

      url.searchParams.append('prompt', params.prompt);

      pending = f(url).then((res) => imageFromResponse(root, res)).finally(() => {
        pending = undefined;
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
  };
}
