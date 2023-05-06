import { MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { DiffusionModel } from '../../../types';

export interface DiffusionModelInputProps {
  model: DiffusionModel;
}

export function DiffusionModelInput(props: DiffusionModelInputProps) {
  const { model } = props;

  return <Stack direction='row' spacing={2}>
    <TextField label='Label' value={model.label} />
    <TextField label='Source' value={model.source} />
    <Select value={model.format} label='Format'>
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
  </Stack>;
}
