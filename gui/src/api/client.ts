import { doesExist } from '@apextoaster/js-utils';

export interface Txt2ImgParams {
  prompt: string;
  cfg: number;
  steps: number;
  width: number;
  height: number;
}

export interface ApiClient {
  txt2img(params: Txt2ImgParams): Promise<string>;
}

export async function imageFromResponse(res: Response) {
  if (res.status === 200) {
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
        console.log('skipping request, one is already pending');
        return pending;
      }

      const url = new URL('/txt2img', root);
      url.searchParams.append('cfg', params.cfg.toFixed(0));
      url.searchParams.append('steps', params.steps.toFixed(0));
      url.searchParams.append('width', params.width.toFixed(0));
      url.searchParams.append('height', params.height.toFixed(0));
      url.searchParams.append('prompt', params.prompt);

      pending = f(url).then((res) => {
        pending = undefined;
        return imageFromResponse(res);
      });

      return await pending;
    },
  }
}