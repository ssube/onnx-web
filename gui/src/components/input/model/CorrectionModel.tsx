import { MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { CorrectionModel } from '../../../types';

export interface CorrectionModelInputProps {
  model: CorrectionModel;

  onChange: (model: CorrectionModel) => void;
}

export function CorrectionModelInput(props: CorrectionModelInputProps) {
  const { model, onChange } = props;

  return <Stack direction='row' spacing={2}>
    <TextField
      value={model.label}
      onChange={(event) => {
        onChange({
          ...model,
          label: event.target.value,
        });
      }}
    />
    <TextField
      value={model.source}
      onChange={(event) => {
        onChange({
          ...model,
          source: event.target.value,
        });
      }}
    />
    <Select
      value={model.format}
      label='Format'
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
    <Select
      value={model.model}
      label='Type'
      onChange={(selection) => {
        onChange({
          ...model,
          model: selection.target.value as 'codeformer',
        });
      }}
    >
      <MenuItem value='codeformer'>Codeformer</MenuItem>
      <MenuItem value='gfpgan'>GFPGAN</MenuItem>
    </Select>
  </Stack>;
}
