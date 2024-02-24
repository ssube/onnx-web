/**
 * Output image size, after upscaling and outscale.
 */
export interface ImageSize {
  width: number;
  height: number;
}

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
  tiled_vae: boolean;
  vae_overlap: number;
  vae_tile: number;
  unet_overlap: number;
  unet_tile: number;

  cfg: number;
  steps: number;
  seed: number;
  eta: number;
}

/**
 * Parameters for txt2img requests.
 */
export type Txt2ImgParams = BaseImgParams & ImageSize;

export interface Img2ImgJSONParams {
  loopback: number;
  sourceFilter: string;
  strength: number;
}

/**
 * Parameters for img2img requests.
 */
export type Img2ImgParams = BaseImgParams & Img2ImgJSONParams & {
  source: Blob;
};

export interface InpaintJSONParams {
  filter: string;
  noise: string;
  strength: number;
  fillColor: string;
  tileOrder: string;
}

/**
 * Parameters for inpaint requests.
 */
export type InpaintParams = BaseImgParams & InpaintJSONParams & {
  mask: Blob;
  source: Blob;
};

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

export interface ExperimentalParams {
  latentSymmetry: {
    enabled: boolean;
    gradientStart: number;
    gradientEnd: number;
    lineOfSymmetry: number;
  };
  promptEditing: {
    enabled: boolean;
    filter: string;
    removeTokens: string;
    addSuffix: string;
    minLength: number;
  };
}
