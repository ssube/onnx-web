import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { DiffusionModel, ModelFormat } from '../../../types.js';

export interface DiffusionModelInputProps {
  key?: number | string;
  model: DiffusionModel;

  onChange: (model: DiffusionModel) => void;
  onRemove: (model: DiffusionModel) => void;
}

export function DiffusionModelInput(props: DiffusionModelInputProps) {
  const { key, model, onChange, onRemove } = props;

  return <Stack direction='row' spacing={2} key={key}>
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
        format: selection.target.value as ModelFormat,
      });
    }}>
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
    <Button onClick={() => onRemove(model)}>Remove</Button>
  </Stack>;
}
