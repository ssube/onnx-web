import { CorrectionModel, DiffusionModel, ExtraNetwork, ExtraSource, ExtrasFile, UpscalingModel } from '../types/model.js';
import { MISSING_INDEX, Slice } from './types.js';

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

// eslint-disable-next-line sonarjs/cognitive-complexity
export function createModelSlice<TState extends ModelSlice>(): Slice<TState, ModelSlice> {
  // eslint-disable-next-line sonarjs/cognitive-complexity
  return (set) => ({
    extras: {
      correction: [],
      diffusion: [],
      networks: [],
      sources: [],
      upscaling: [],
    },
    setExtras(extras) {
      set((prev) => ({
        ...prev,
        extras: {
          ...prev.extras,
          ...extras,
        },
      }));
    },
    setCorrectionModel(model) {
      set((prev) => {
        const correction = [...prev.extras.correction];
        const exists = correction.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          correction.push(model);
        } else {
          correction[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            correction,
          },
        };
      });
    },
    setDiffusionModel(model) {
      set((prev) => {
        const diffusion = [...prev.extras.diffusion];
        const exists = diffusion.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          diffusion.push(model);
        } else {
          diffusion[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            diffusion,
          },
        };
      });
    },
    setExtraNetwork(model) {
      set((prev) => {
        const networks = [...prev.extras.networks];
        const exists = networks.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          networks.push(model);
        } else {
          networks[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            networks,
          },
        };
      });
    },
    setExtraSource(model) {
      set((prev) => {
        const sources = [...prev.extras.sources];
        const exists = sources.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          sources.push(model);
        } else {
          sources[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            sources,
          },
        };
      });
    },
    setUpscalingModel(model) {
      set((prev) => {
        const upscaling = [...prev.extras.upscaling];
        const exists = upscaling.findIndex((it) => model.name === it.name);
        if (exists === MISSING_INDEX) {
          upscaling.push(model);
        } else {
          upscaling[exists] = model;
        }

        return {
          ...prev,
          extras: {
            ...prev.extras,
            upscaling,
          },
        };
      });
    },
    removeCorrectionModel(model) {
      set((prev) => {
        const correction = prev.extras.correction.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            correction,
          },
        };
      });

    },
    removeDiffusionModel(model) {
      set((prev) => {
        const diffusion = prev.extras.diffusion.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            diffusion,
          },
        };
      });

    },
    removeExtraNetwork(model) {
      set((prev) => {
        const networks = prev.extras.networks.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            networks,
          },
        };
      });

    },
    removeExtraSource(model) {
      set((prev) => {
        const sources = prev.extras.sources.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            sources,
          },
        };
      });

    },
    removeUpscalingModel(model) {
      set((prev) => {
        const upscaling = prev.extras.upscaling.filter((it) => model.name !== it.name);;
        return {
          ...prev,
          extras: {
            ...prev.extras,
            upscaling,
          },
        };
      });
    },
  });
}
