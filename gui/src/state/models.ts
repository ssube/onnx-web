import { CorrectionModel, DiffusionModel, ExtraNetwork, ExtraSource, ExtrasFile, UpscalingModel } from '../types/model.js';

export interface ModelSlice {
  extras: ExtrasFile;

  removeCorrectionModel(model: CorrectionModel): void;
  removeDiffusionModel(model: DiffusionModel): void;
  removeExtraNetwork(model: ExtraNetwork): void;
  removeExtraSource(model: ExtraSource): void;
  removeUpscalingModel(model: UpscalingModel): void;

  setExtras(extras: Partial<ExtrasFile>): void;

  setCorrectionModel(model: CorrectionModel): void;
  setDiffusionModel(model: DiffusionModel): void;
  setExtraNetwork(model: ExtraNetwork): void;
  setExtraSource(model: ExtraSource): void;
  setUpscalingModel(model: UpscalingModel): void;
}
