import { ServerParams } from '../config.js';
import { ExtrasFile } from '../types.js';

/**
 * Shared parameters for anything using models, which is pretty much everything.
 */
export interface ModelParams {
  /**
   * The diffusion model to use.
   */
  model: string;

  /**
   * Specialized pipeline to use.
   */
  pipeline: string;

  /**
   * The hardware acceleration platform to use.
   */
  platform: string;

  /**
   * The upscaling model to use.
   */
  upscaling: string;

  /**
   * The correction model to use.
   */
  correction: string;

  /**
   * ControlNet to be used.
   */
  control: string;
}

/**
 * Shared parameters for most of the image requests.
 */
export interface BaseImgParams {
  scheduler: string;
  prompt: string;
  negativePrompt?: string;

  batch: number;
  tiledVAE: boolean;
  tiles: number;
  overlap: number;
  stride: number;

  cfg: number;
  steps: number;
  seed: number;
  eta: number;
}

/**
 * Parameters for txt2img requests.
 */
export interface Txt2ImgParams extends BaseImgParams {
  width: number;
  height: number;
}

/**
 * Parameters for img2img requests.
 */
export interface Img2ImgParams extends BaseImgParams {
  source: Blob;

  loopback: number;
  sourceFilter: string;
  strength: number;
}

/**
 * Parameters for inpaint requests.
 */
export interface InpaintParams extends BaseImgParams {
  mask: Blob;
  source: Blob;

  filter: string;
  noise: string;
  strength: number;
  fillColor: string;
  tileOrder: string;
}

/**
 * Additional parameters for outpaint border.
 *
 * @todo should be nested under inpaint/outpaint params
 */
export interface OutpaintPixels {
  enabled: boolean;

  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * Parameters for outpaint requests.
 */
export type OutpaintParams = InpaintParams & OutpaintPixels;

/**
 * Additional parameters for the inpaint brush.
 *
 * These are not currently sent to the server and only stored in state.
 *
 * @todo move to state
 */
export interface BrushParams {
  color: number;
  size: number;
  strength: number;
}

/**
 * Additional parameters for upscaling. May be sent with most other requests to run a post-pipeline.
 */
export interface UpscaleParams {
  enabled: boolean;
  upscaleOrder: string;

  denoise: number;
  scale: number;
  outscale: number;

  faces: boolean;
  faceStrength: number;
  faceOutscale: number;
}

/**
 * Parameters for upscale requests.
 */
export interface UpscaleReqParams extends BaseImgParams {
  source: Blob;
}

/**
 * Parameters for blend requests.
 */
export interface BlendParams {
  sources: Array<Blob>;
  mask: Blob;
}

export interface HighresParams {
  enabled: boolean;

  highresIterations: number;
  highresMethod: string;
  highresScale: number;
  highresSteps: number;
  highresStrength: number;
}

export interface Txt2ImgStage {
  name: string;
  type: 'source-txt2img';
  params: Txt2ImgParams;
}

export interface Img2ImgStage {
  name: string;
  type: 'blend-img2img';
  params: Img2ImgParams;
}

export interface GridStage {
  name: string;
  type: 'blend-grid';
  params: {
    height: number;
    width: number;
  };
}

export interface OutputStage {
  name: string;
  type: 'persist-disk';
  params: {
    /* none */
  };
}

export interface ChainPipeline {
  stages: Array<Txt2ImgStage | Img2ImgStage | GridStage | OutputStage>;
}

/**
 * Output image data within the response.
 */
export interface ImageOutput {
  key: string;
  url: string;
}

/**
 * Output image size, after upscaling and outscale.
 */
export interface ImageSize {
  width: number;
  height: number;
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

export interface ApiClient {
  extras(): Promise<ExtrasFile>;

  writeExtras(extras: ExtrasFile): Promise<WriteExtrasResponse>;

  /**
   * List the available filter masks for inpaint.
   */
  filters(): Promise<FilterResponse>;

  /**
   * List the available models.
   */
  models(): Promise<ModelResponse>;

  /**
   * List the available noise sources for inpaint.
   */
  noises(): Promise<Array<string>>;

  /**
   * Get the valid server parameters to validate image parameters.
   */
  params(): Promise<ServerParams>;

  /**
   * Get the available pipelines.
   */
  pipelines(): Promise<Array<string>>;

  /**
   * Get the available hardware acceleration platforms.
   */
  platforms(): Promise<Array<string>>;

  /**
   * List the available pipeline schedulers.
   */
  schedulers(): Promise<Array<string>>;

  /**
   * Load extra strings from the server.
   */
  strings(): Promise<Record<string, {
    translation: Record<string, string>;
  }>>;

  /**
   * Start a txt2img pipeline.
   */
  txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an im2img pipeline.
   */
  img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an inpaint pipeline.
   */
  inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an outpaint pipeline.
   */
  outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry>;

  /**
   * Start an upscale pipeline.
   */
  upscale(model: ModelParams, params: UpscaleReqParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<ImageResponseWithRetry>;

  /**
   * Start a blending pipeline.
   */
  blend(model: ModelParams, params: BlendParams, upscale?: UpscaleParams): Promise<ImageResponseWithRetry>;

  chain(model: ModelParams, chain: ChainPipeline): Promise<ImageResponse>;

  /**
   * Check whether job has finished and its output is ready.
   */
  ready(key: string): Promise<ReadyResponse>;

  /**
   * Cancel an existing job.
   */
  cancel(key: string): Promise<boolean>;

  /**
   * Retry a previous job using the same parameters.
   */
  retry(params: RetryParams): Promise<ImageResponseWithRetry>;

  /**
   * Restart the image job workers.
   */
  restart(): Promise<boolean>;

  /**
   * Check the status of the image job workers.
   */
  status(): Promise<Array<unknown>>;
}
