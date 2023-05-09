import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { AnyFormat, ExtraSource } from '../../../types.js';

export interface ExtraSourceInputProps {
  key?: number | string;
  model: ExtraSource;

  onChange: (model: ExtraSource) => void;
  onRemove: (model: ExtraSource) => void;
}

export function ExtraSourceInput(props: ExtraSourceInputProps) {
  const { key, model, onChange, onRemove } = props;

  return <Stack direction='row' spacing={2} key={key}>
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
          format: selection.target.value as AnyFormat,
        });
      }}
    >
      <MenuItem value='ckpt'>ckpt</MenuItem>
      <MenuItem value='safetensors'>safetensors</MenuItem>
      <MenuItem value='json'>json</MenuItem>
      <MenuItem value='yaml'>yaml</MenuItem>
    </Select>
    <TextField label='Folder' value={model.dest} onChange={(event) => {
      onChange({
        ...model,
        dest: event.target.value,
      });
    }} />
    <Button onClick={() => onRemove(model)}>Remove</Button>
  </Stack>;
}
