import { MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { DiffusionModel } from '../../../types.js';

export interface DiffusionModelInputProps {
  model: DiffusionModel;

  onChange: (model: DiffusionModel) => void;
}

export function DiffusionModelInput(props: DiffusionModelInputProps) {
  const { model, onChange } = props;

  return <Stack direction='row' spacing={2}>
    <TextField label='Label' value={model.label} onChange={(event) => {
      onChange({
        ...model,
        label: event.target.value,
      });
    }} />
    <TextField label='Source' value={model.source} onChange={(event) => {
      onChange({
        ...model,
        source: event.target.value,
      });
    }} />
    <Select value={model.format} label='Format' onChange={(selection) => {
      onChange({
        ...model,
        format: selection.target.value as 'ckpt' | 'safetensors',
      });
    }}>
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
  </Stack>;
}
