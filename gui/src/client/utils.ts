import { doesExist } from '@apextoaster/js-utils';
import { HighresParams, ModelParams, Txt2ImgParams, UpscaleParams } from '../types/params.js';
import { ChainPipeline, ChainStageParams, STRING_PARAMETERS } from '../types/chain.js';

export interface PipelineVariable {
  parameter: 'prompt' | 'cfg' | 'seed' | 'steps' | 'eta' | 'scheduler' | 'token';
  value: string;
}

export interface PipelineGrid {
  enabled: boolean;
  columns: PipelineVariable;
  rows: PipelineVariable;
}

export const EXPR_STRICT_NUMBER = /^-?\d+$/;
export const EXPR_NUMBER_RANGE = /^(-?\d+)-(-?\d+)$/;

export const MAX_SEED_SIZE = 32;
export const MAX_SEED = (2**MAX_SEED_SIZE) - 1;

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

export function newSeed(): number {
  return Math.floor(Math.random() * MAX_SEED);
}

export function replaceRandomSeeds(key: string, values: Array<number | string>): Array<number | string> {
  if (key !== 'seed') {
    return values;
  }

  return values.map((it) => {
    // eslint-disable-next-line @typescript-eslint/no-magic-numbers
    if (it === '-1' || it === -1) {
      return newSeed();
    }

    return it;
  });
}

export function rangeSplit(parameter: string, value: string): Array<number | string> {
  const csv = value.split(',').map((it) => it.trim());

  if (STRING_PARAMETERS.includes(parameter)) {
    return csv;
  }

  return csv.flatMap((it) => expandRanges(it));
}

export function expandRanges(range: string): Array<string | number> {
  if (EXPR_STRICT_NUMBER.test(range)) {
    // entirely numeric, return after parsing
    const val = parseInt(range, 10);
    return [val];
  }

  if (EXPR_NUMBER_RANGE.test(range)) {
    const match = EXPR_NUMBER_RANGE.exec(range);
    if (doesExist(match)) {
      const [_full, startStr, endStr] = Array.from(match);
      const start = parseInt(startStr, 10);
      const end = parseInt(endStr, 10);

      return new Array(end - start).fill(0).map((_value, idx) => idx + start);
    }
  }

  return [];
}

export const GRID_TILE_SIZE = 8192;

// eslint-disable-next-line max-params
export function makeTxt2ImgGridPipeline(grid: PipelineGrid, model: ModelParams, params: Txt2ImgParams, upscale?: UpscaleParams, highres?: HighresParams): ChainPipeline {
  const pipeline: ChainPipeline = {
    defaults: {
      ...model,
      ...params,
    },
    stages: [],
  };

  const tiles: ChainStageParams = {
    tiles: GRID_TILE_SIZE,
  };

  const rows = replaceRandomSeeds(grid.rows.parameter, rangeSplit(grid.rows.parameter, grid.rows.value));
  const columns = replaceRandomSeeds(grid.columns.parameter, rangeSplit(grid.columns.parameter, grid.columns.value));

  let i = 0;

  for (const row of rows) {
    for (const column of columns) {
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
      height: rows.length,
      width: columns.length,
    },
  });

  pipeline.stages.push({
    name: 'save',
    type: 'persist-disk',
    params: tiles,
  });

  return pipeline;
}
