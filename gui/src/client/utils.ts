import { ChainPipeline, HighresParams, ModelParams, Txt2ImgParams, UpscaleParams } from './types.js';

export interface PipelineVariable {
  parameter: 'prompt' | 'cfg' | 'seed' | 'steps';
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
    },
  });

  return pipeline;
}
