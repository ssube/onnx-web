import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControl, InputLabel, MenuItem, Select, Stack, TextField } from '@mui/material';
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

  return <Stack direction='column' spacing={2}>
    <Stack direction='row' spacing={2}>
      <InputLabel>Grid Mode</InputLabel>
      <Checkbox checked={grid.enabled} onChange={() => props.setGrid({
        enabled: grid.enabled === false,
      })} />
    </Stack>
    <Stack direction='row' spacing={2}>
      <FormControl>
        <InputLabel id='TODO'>Columns</InputLabel>
        <Select onChange={(event) => props.setGrid({
          columns: {
            parameter: event.target.value as VariableKey,
            input: '',
            values: [],
          },
        })} value={grid.columns.parameter}>
          <MenuItem key='prompt' value='prompt'>Prompt</MenuItem>
          <MenuItem key='seed' value='seed'>Seed</MenuItem>
          <MenuItem key='steps' value='steps'>Steps</MenuItem>
        </Select>
      </FormControl>
      <TextField label={grid.columns.parameter} value={grid.columns.input} onChange={(event) => props.setGrid({
        columns: {
          parameter: grid.columns.parameter,
          input: event.target.value,
          values: rangeSplit(grid.columns.parameter, event.target.value),
        },
      })} />
    </Stack>
    <Stack direction='row' spacing={2}>
      <FormControl>
        <InputLabel id='TODO'>Rows</InputLabel>
        <Select onChange={(event) => props.setGrid({
          rows: {
            parameter: event.target.value as VariableKey,
            input: '',
            values: [],
          }
        })} value={grid.rows.parameter}>
          <MenuItem key='prompt' value='prompt'>Prompt</MenuItem>
          <MenuItem key='seed' value='seed'>Seed</MenuItem>
          <MenuItem key='steps' value='steps'>Steps</MenuItem>
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
  </Stack>;
}

export function rangeSplit(parameter: string, value: string): Array<number | string> {
  // string values
  if (parameter === 'prompt') {
    return value.split('\n');
  }

  return value.split(',').map((it) => it.trim()).flatMap((it) => expandRanges(it));
}

export const EXPR_STRICT_NUMBER = /^[0-9]+$/;
export const EXPR_NUMBER_RANGE = /^([0-9]+)-([0-9]+)$/;

export function expandRanges(range: string): Array<string | number> {
  if (EXPR_STRICT_NUMBER.test(range)) {
    // entirely numeric, return without parsing
    return [parseInt(range, 10)];
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
