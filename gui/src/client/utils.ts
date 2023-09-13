import { doesExist } from '@apextoaster/js-utils';
import { HighresParams, ModelParams, Txt2ImgParams, UpscaleParams } from '../types/params.js';
import { ChainPipeline } from '../types/chain.js';

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

export function replacePromptTokens(grid: PipelineGrid, params: Txt2ImgParams, columnValue: string | number, rowValue: string | number): {prompt: string} {
  const result = {
    negativePrompt: params.negativePrompt,
    prompt: params.prompt,
  };

  if (grid.columns.parameter === 'token') {
    result.prompt = result.prompt.replace('__column__', columnValue.toString());

    if (doesExist(result.negativePrompt)) {
      result.negativePrompt = result.negativePrompt.replace('__column__', columnValue.toString());
    }
  }

  if (grid.rows.parameter === 'token') {
    result.prompt = result.prompt.replace('__row__', rowValue.toString());

    if (doesExist(result.negativePrompt)) {
      result.negativePrompt = result.negativePrompt.replace('__row__', rowValue.toString());
    }
  }

  return result;
}

// eslint-disable-next-line max-params
export function makeTxt2ImgGridPipeline(grid: PipelineGrid, model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): ChainPipeline {
  const pipeline: ChainPipeline = {
    defaults: {
      ...model,
      ...params,
    },
    stages: [],
  };

  const tiles = {
    // eslint-disable-next-line camelcase
    tile_size: 8192,
  };

  let i = 0;

  for (const row of grid.rows.values) {
    for (const column of grid.columns.values) {
      const prompt = replacePromptTokens(grid, params, column, row);

      pipeline.stages.push({
        name: `cell-${i}`,
        type: 'source-txt2img',
        params: {
          ...params,
          ...prompt,
          ...model,
          ...tiles,
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
      ...tiles,
      height: grid.rows.values.length,
      width: grid.columns.values.length,
    },
  });

  pipeline.stages.push({
    name: 'save',
    type: 'persist-disk',
    params: tiles,
  });

  return pipeline;
}
