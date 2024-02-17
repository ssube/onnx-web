import { Maybe } from '@apextoaster/js-utils';
import { ServerParams } from '../config.js';
import { JobResponse, JobResponseWithRetry, SuccessJobResponse } from '../types/api-v2.js';
import { FilterResponse, ModelResponse, RetryParams, WriteExtrasResponse } from '../types/api.js';
import { ChainPipeline } from '../types/chain.js';
import { ExtrasFile } from '../types/model.js';
import {
  BlendParams,
  ExperimentalParams,
  HighresParams,
  Img2ImgParams,
  InpaintParams,
  ModelParams,
  OutpaintParams,
  Txt2ImgParams,
  UpscaleParams,
  UpscaleReqParams,
} from '../types/params.js';

export interface ApiClient {
  /**
   * Get the first extras file.
   */
  extras(): Promise<ExtrasFile>;

  /**
   * Update the first extras file.
   */
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
   * Get the available wildcards.
   */
  wildcards(): Promise<Array<string>>;

  /**
   * Start a txt2img pipeline.
   */
  txt2img(model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry>;

  /**
   * Start an im2img pipeline.
   */
  img2img(model: ModelParams, params: Img2ImgParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry>;

  /**
   * Start an inpaint pipeline.
   */
  inpaint(model: ModelParams, params: InpaintParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry>;

  /**
   * Start an outpaint pipeline.
   */
  outpaint(model: ModelParams, params: OutpaintParams, upscale?: UpscaleParams, highres?: HighresParams, experimental?: ExperimentalParams): Promise<JobResponseWithRetry>;

  /**
   * Start an upscale pipeline.
   */
  upscale(model: ModelParams, params: UpscaleReqParams, upscale?: UpscaleParams, highres?: HighresParams): Promise<JobResponseWithRetry>;

  /**
   * Start a blending pipeline.
   */
  blend(model: ModelParams, params: BlendParams, upscale?: UpscaleParams): Promise<JobResponseWithRetry>;

  /**
   * Start a custom chain pipeline.
   */
  chain(model: ModelParams, chain: ChainPipeline): Promise<JobResponse>;

  /**
   * Check whether job has finished and its output is ready.
   */
  status(keys: Array<string>): Promise<Array<JobResponse>>;

  /**
   * Cancel an existing job.
   */
  cancel(keys: Array<string>): Promise<Array<JobResponse>>;

  /**
   * Retry a previous job using the same parameters.
   */
  retry(params: RetryParams): Promise<JobResponseWithRetry>;

  /**
   * Restart the image job workers.
   */
  restart(): Promise<boolean>;

  /**
   * Check the status of the image job workers.
   */
  workers(): Promise<Array<unknown>>;

  outputURL(image: SuccessJobResponse, index: number): string;

  thumbnailURL(image: SuccessJobResponse, index: number): Maybe<string>;
}
