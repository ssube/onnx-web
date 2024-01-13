import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { PipelineGrid } from '../../client/utils.js';
import { OnnxState, StateContext } from '../../state/full.js';
import { VARIABLE_PARAMETERS } from '../../types/chain.js';
import { STANDARD_SPACING } from '../../constants.js';

export interface VariableControlProps {
  selectGrid: (state: OnnxState) => PipelineGrid;
  setGrid: (grid: Partial<PipelineGrid>) => void;
}

export type VariableKey = 'prompt' | 'steps' | 'seed';

export function VariableControl(props: VariableControlProps) {
  const store = mustExist(useContext(StateContext));
  const grid = useStore(store, props.selectGrid);

  const stack = [
    <Stack direction='row' spacing={STANDARD_SPACING} key='variable-enable'>
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
      <Stack direction='row' spacing={STANDARD_SPACING} key='variable-row'>
        <FormControl>
          <InputLabel id='TODO'>Columns</InputLabel>
          <Select onChange={(event) => props.setGrid({
            columns: {
              parameter: event.target.value as VariableKey,
              value: '',
            },
          })} value={grid.columns.parameter}>
            {...parameterList([grid.rows.parameter])}
          </Select>
        </FormControl>
        <TextField label={grid.columns.parameter} value={grid.columns.value} onChange={(event) => props.setGrid({
          columns: {
            parameter: grid.columns.parameter,
            value: event.target.value,
          },
        })} />
      </Stack>,
      <Stack direction='row' spacing={STANDARD_SPACING} key='variable-column'>
        <FormControl>
          <InputLabel id='TODO'>Rows</InputLabel>
          <Select onChange={(event) => props.setGrid({
            rows: {
              parameter: event.target.value as VariableKey,
              value: '',
            }
          })} value={grid.rows.parameter}>
            {...parameterList([grid.columns.parameter])}
          </Select>
        </FormControl>
        <TextField label={grid.rows.parameter} value={grid.rows.value} onChange={(event) => props.setGrid({
          rows: {
            parameter: grid.rows.parameter,
            value: event.target.value,
          }
        })} />
      </Stack>
    );
  }

  return <Stack direction='column' spacing={STANDARD_SPACING}>{...stack}</Stack>;
}

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
