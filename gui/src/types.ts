export type TorchFormat = 'bin' | 'ckpt' | 'pt' | 'pth';
export type OnnxFormat = 'onnx';
export type SafetensorFormat = 'safetensors';

export interface BaseModel {
  /**
   * Format of the model, used when downloading files that may not have a format in their URL.
   */
  format: OnnxFormat | SafetensorFormat | TorchFormat;

  /**
   * Localized label of the model.
   */
  label: string;

  /**
   * Filename of the model.
   */
  name: string;

  /**
   * Source URL or local path.
   */
  source: string;
}

export interface DiffusionModel extends BaseModel {
  config?: string;
  image_size?: string;
  inversions?: Array<unknown>;
  loras?: Array<unknown>;
  pipeline?: string;
  vae?: string;
  version?: string;
}

export interface UpscalingModel extends BaseModel {
  model?: 'bsrgan' | 'resrgan' | 'swinir';
  scale: number;
}

export interface CorrectionModel extends BaseModel {
  model?: 'codeformer' | 'gfpgan';
}

export interface ExtraNetwork extends BaseModel {
  model: 'concept' | 'embeddings' | 'cloneofsimo' | 'sd-scripts';
  type: 'inversion' | 'lora';
}

export interface ExtraSource {
  dest?: string;
  format?: string;
  name: string;
  source: string;
}

export interface ExtrasFile {
  correction: Array<CorrectionModel>;
  diffusion: Array<DiffusionModel>;
  upscaling: Array<UpscalingModel>;
  networks: Array<ExtraNetwork>;
  sources: Array<ExtraSource>;
}
