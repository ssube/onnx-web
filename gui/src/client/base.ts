import { ServerParams } from '../config.js';
import { ExtrasFile } from '../types/model.js';
import { WriteExtrasResponse, FilterResponse, ModelResponse, ImageResponseWithRetry, ImageResponse, ReadyResponse, RetryParams } from '../types/api.js';
import { ChainPipeline } from '../types/chain.js';
import { ModelParams, Txt2ImgParams, UpscaleParams, HighresParams, Img2ImgParams, InpaintParams, OutpaintParams, UpscaleReqParams, BlendParams } from '../types/params.js';

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
