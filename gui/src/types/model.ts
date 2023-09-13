export type TorchFormat = 'bin' | 'ckpt' | 'pt' | 'pth';
export type OnnxFormat = 'onnx';
export type SafetensorFormat = 'safetensors';
export type TensorFormat = TorchFormat | SafetensorFormat;
export type ModelFormat = TensorFormat | OnnxFormat;
export type MarkupFormat = 'json' | 'yaml';
export type AnyFormat = MarkupFormat | ModelFormat;

export type UpscalingArch = 'bsrgan' | 'resrgan' | 'swinir';
export type CorrectionArch = 'codeformer' | 'gfpgan';

export type NetworkType = 'inversion' | 'lora';
export type NetworkModel = 'concept' | 'embeddings' | 'cloneofsimo' | 'sd-scripts';

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
  model?: UpscalingArch;
  scale: number;
}

export interface CorrectionModel extends BaseModel {
  model?: CorrectionArch;
}

export interface ExtraNetwork extends BaseModel {
  model: NetworkModel;
  type: NetworkType;
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

export type DeepPartial<T> = T extends object ? {
  [P in keyof T]?: DeepPartial<T[P]>;
} : T;
