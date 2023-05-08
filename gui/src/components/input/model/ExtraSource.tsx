import * as React from 'react';
import { MenuItem, Select, Stack, TextField } from '@mui/material';

import { ExtraSource } from '../../../types';

export interface ExtraSourceInputProps {
  model: ExtraSource;

  onChange: (model: ExtraSource) => void;
}

export function ExtraSourceInput(props: ExtraSourceInputProps) {
  const { model, onChange } = props;

  return <Stack direction='row' spacing={2}>
    <TextField label='Name' value={model.name} onChange={(event) => {
      onChange({
        ...model,
        name: event.target.value,
      });
    }} />
    <TextField label='Source' value={model.source} onChange={(event) => {
      onChange({
        ...model,
        source: event.target.value,
      });
    }} />
    <Select
      label='Format'
      value={model.format}
      onChange={(selection) => {
        onChange({
          ...model,
          format: selection.target.value as 'safetensors',
        });
      }}
    >
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
    </Select>
    <TextField label='Folder' value={model.dest} onChange={(event) => {
      onChange({
        ...model,
        dest: event.target.value,
      });
    }} />
  </Stack>;
}
