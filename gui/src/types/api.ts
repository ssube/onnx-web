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
 *
 * @deprecated
 */
export interface ImageOutput {
  key: string;
  url: string;
}

/**
 * General response for most image requests.
 *
 * @deprecated
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

export interface ControlNetwork {
  name: string;
  type: 'control';
}

export interface EmbeddingNetwork {
  label: string;
  name: string;
  token: string;
  type: 'inversion';
  // TODO: add layer count
}

export interface LoraNetwork {
  name: string;
  label: string;
  tokens: Array<string>;
  type: 'lora';
}

export type NetworkModel = EmbeddingNetwork | LoraNetwork | ControlNetwork;

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

/**
 * Status response from the image endpoint, with parameters to retry the job if it fails.
 *
 * @deprecated
 */
export interface ImageResponseWithRetry {
  image: ImageResponse;
  retry: RetryParams;
}

/**
 * @deprecated
 */
export interface ImageMetadata {
  highres: HighresParams;
  outputs: string | Array<string>;
  params: Txt2ImgParams | Img2ImgParams | InpaintParams;
  upscale: UpscaleParams;

  input_size: ImageSize;
  size: ImageSize;
}
