import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { PipelineGrid } from '../../client/utils.js';
import { OnnxState, StateContext } from '../../state.js';

export interface VariableControlProps {
  selectGrid: (state: OnnxState) => PipelineGrid;
  setGrid: (grid: Partial<PipelineGrid>) => void;
}

export type VariableKey = 'prompt' | 'steps' | 'seed';

export function VariableControl(props: VariableControlProps) {
  const store = mustExist(useContext(StateContext));
  const grid = useStore(store, props.selectGrid);

  const stack = [
    <Stack direction='row' spacing={2} key='variable-enable'>
      <FormControl>
        <FormControlLabel
          label='Grid Mode'
          control={<Checkbox
            checked={grid.enabled}
            onChange={() => props.setGrid({
              enabled: grid.enabled === false,
            })}
          />}
        />
      </FormControl>
    </Stack>,
  ];

  if (grid.enabled) {
    stack.push(
      <Stack direction='row' spacing={2} key='variable-row'>
        <FormControl>
          <InputLabel id='TODO'>Columns</InputLabel>
          <Select onChange={(event) => props.setGrid({
            columns: {
              parameter: event.target.value as VariableKey,
              input: '',
              values: [],
            },
          })} value={grid.columns.parameter}>
            {...parameterList([grid.rows.parameter])}
          </Select>
        </FormControl>
        <TextField label={grid.columns.parameter} value={grid.columns.input} onChange={(event) => props.setGrid({
          columns: {
            parameter: grid.columns.parameter,
            input: event.target.value,
            values: rangeSplit(grid.columns.parameter, event.target.value),
          },
        })} />
      </Stack>,
      <Stack direction='row' spacing={2} key='variable-column'>
        <FormControl>
          <InputLabel id='TODO'>Rows</InputLabel>
          <Select onChange={(event) => props.setGrid({
            rows: {
              parameter: event.target.value as VariableKey,
              input: '',
              values: [],
            }
          })} value={grid.rows.parameter}>
            {...parameterList([grid.columns.parameter])}
          </Select>
        </FormControl>
        <TextField label={grid.rows.parameter} value={grid.rows.input} onChange={(event) => props.setGrid({
          rows: {
            parameter: grid.rows.parameter,
            input: event.target.value,
            values: rangeSplit(grid.rows.parameter, event.target.value),
          }
        })} />
      </Stack>
    );
  }

  return <Stack direction='column' spacing={2}>{...stack}</Stack>;
}

export function rangeSplit(parameter: string, value: string): Array<number | string> {
  const csv = value.split(',').map((it) => it.trim());

  if (STRING_PARAMETERS.includes(parameter)) {
    return csv;
  }

  return csv.flatMap((it) => expandRanges(it));
}

export const EXPR_STRICT_NUMBER = /^-?\d+$/;
export const EXPR_NUMBER_RANGE = /^(\d+)-(\d+)$/;

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

export const VARIABLE_PARAMETERS = ['prompt', 'negativePrompt', 'seed', 'steps', 'cfg', 'scheduler', 'eta', 'token'];
export const STRING_PARAMETERS = ['prompt', 'negativePrompt', 'scheduler', 'token'];

export function parameterList(exclude?: Array<string>) {
  const items = [];

  for (const variable of VARIABLE_PARAMETERS) {
    if (variable !== 'token' && doesExist(exclude) && exclude.includes(variable)) {
      continue;
    }

    items.push(<MenuItem key={variable} value={variable}>{variable}</MenuItem>);
  }

  return items;
}
