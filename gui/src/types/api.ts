import {
  BlendParams,
  HighresParams,
  Img2ImgParams,
  InpaintParams,
  ModelParams,
  OutpaintParams,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
} from './params.js';

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
