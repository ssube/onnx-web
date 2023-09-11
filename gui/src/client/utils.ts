import { doesExist } from '@apextoaster/js-utils';
import { ChainPipeline, HighresParams, ModelParams, Txt2ImgParams, UpscaleParams } from './types.js';

export interface PipelineVariable {
  parameter: 'prompt' | 'cfg' | 'seed' | 'steps' | 'eta' | 'scheduler' | 'token';
  input: string;
  values: Array<number | string>;
}

export interface PipelineGrid {
  enabled: boolean;
  columns: PipelineVariable;
  rows: PipelineVariable;
}

// eslint-disable-next-line max-params
export function buildPipelineForTxt2ImgGrid(grid: PipelineGrid, model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): ChainPipeline {
  const pipeline: ChainPipeline = {
    stages: [],
  };

  let i = 0;

  for (const column of grid.columns.values) {
    for (const row of grid.rows.values) {
      pipeline.stages.push({
        name: `cell-${i}`,
        type: 'source-txt2img',
        params: {
          ...params,
          ...model,
          [grid.columns.parameter]: column,
          [grid.rows.parameter]: row,
          // eslint-disable-next-line camelcase
          tile_size: 8192,
        },
      });

      i += 1;
    }
  }

  pipeline.stages.push({
    name: 'grid',
    type: 'blend-grid',
    params: {
      ...params,
      ...model,
      height: grid.rows.values.length,
      width: grid.columns.values.length,
      // eslint-disable-next-line camelcase
      tile_size: 8192,
    },
  });

  pipeline.stages.push({
    name: 'save',
    type: 'persist-disk',
    params: {
      // eslint-disable-next-line camelcase
      tile_size: 8192,
    },
  });

  return pipeline;
}
