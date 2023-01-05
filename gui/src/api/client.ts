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
  params: Txt2ImgParams;
  path: string;
}

export interface ApiClient {
  txt2img(params: Txt2ImgParams): Promise<string>;
}

export const STATUS_SUCCESS = 200;

export async function imageFromResponse(res: Response) {
  if (res.status === STATUS_SUCCESS) {
    const imageBlob = await res.blob();
    return URL.createObjectURL(imageBlob);
  } else {
    throw new Error('request error');
  }
}

export function makeClient(root: string, f = fetch): ApiClient {
  let pending: Promise<string> | undefined;

  return {
    async txt2img(params: Txt2ImgParams): Promise<string> {
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

      pending = f(url).then((res) => imageFromResponse(res)).finally(() => {
        pending = undefined;
      });

      // eslint-disable-next-line no-return-await
      return await pending;
    },
  };
}
