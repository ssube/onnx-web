export interface Txt2ImgParams {
  prompt: string;
  cfg: number;
  steps: number;
}

export interface ApiClient {
  txt2img(params: Txt2ImgParams): Promise<string>;
}

export function makeClient(root: string, f = fetch): ApiClient {
  return {
    async txt2img(params: Txt2ImgParams): Promise<string> {
      const { prompt, cfg, steps } = params;

      const safePrompt = encodeURIComponent(prompt);
      const url = `${root}/txt2img?prompt=${safePrompt}&steps=${steps.toFixed(0)}&cfg=${cfg.toFixed(0)}`;
      const res = await f(url);

      if (res.status === 200) {
        const imageBlob = await res.blob();
        return URL.createObjectURL(imageBlob);
      } else {
        throw new Error('request error');
      }
    },
  }
}