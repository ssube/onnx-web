import { ImageSize, Img2ImgParams, ModelParams, Txt2ImgParams } from './params.js';

export interface ChainStageParams {
  tiles: number;
}

export interface Txt2ImgStage {
  name: string;
  type: 'source-txt2img';
  params: Partial<Txt2ImgParams & ChainStageParams>;
}

export interface Img2ImgStage {
  name: string;
  type: 'blend-img2img';
  params: Partial<Img2ImgParams & ChainStageParams>;
}

export interface GridStage {
  name: string;
  type: 'blend-grid';
  params: Partial<ImageSize & ChainStageParams>;
}

export interface OutputStage {
  name: string;
  type: 'persist-disk';
  params: Partial<ChainStageParams>;
}

export interface ChainPipeline {
  /* defaults?: {
    txt2img?: Txt2ImgParams;
    img2img?: Img2ImgParams;
  }; */

  defaults?: Txt2ImgParams & ModelParams;

  stages: Array<Txt2ImgStage | Img2ImgStage | GridStage | OutputStage>;
}
