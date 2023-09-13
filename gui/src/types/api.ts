import {
  BaseImgParams,
  ModelParams,
  Txt2ImgParams,
  UpscaleParams,
  HighresParams,
  Img2ImgParams,
  InpaintParams,
  OutpaintParams,
  UpscaleReqParams,
  BlendParams,
  ImageSize,
} from './params.js';

/**
 * Output image data within the response.
 */
export interface ImageOutput {
  key: string;
  url: string;
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
  type: 'control' | 'inversion' | 'lora';
  // TODO: add token
  // TODO: add layer/token count
}

export interface FilterResponse {
  mask: Array<string>;
  source: Array<string>;
}

/**
 * List of available models.
 */
export interface ModelResponse {
  correction: Array<string>;
  diffusion: Array<string>;
  networks: Array<NetworkModel>;
  upscaling: Array<string>;
}

export interface WriteExtrasResponse {
  file: string;
  successful: Array<string>;
  errors: Array<string>;
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
  highres?: HighresParams;
} | {
  type: 'inpaint';
  model: ModelParams;
  params: InpaintParams;
  upscale?: UpscaleParams;
  highres?: HighresParams;
} | {
  type: 'outpaint';
  model: ModelParams;
  params: OutpaintParams;
  upscale?: UpscaleParams;
  highres?: HighresParams;
} | {
  type: 'upscale';
  model: ModelParams;
  params: UpscaleReqParams;
  upscale?: UpscaleParams;
  highres?: HighresParams;
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

export interface ImageMetadata {
  highres: HighresParams;
  outputs: string | Array<string>;
  params: Txt2ImgParams | Img2ImgParams | InpaintParams;
  upscale: UpscaleParams;

  input_size: ImageSize;
  size: ImageSize;
}
