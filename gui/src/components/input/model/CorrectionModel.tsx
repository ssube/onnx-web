import { Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { CorrectionArch, CorrectionModel, ModelFormat } from '../../../types.js';

export interface CorrectionModelInputProps {
  key?: number | string;
  model: CorrectionModel;

  onChange: (model: CorrectionModel) => void;
  onRemove: (model: CorrectionModel) => void;
}

export function CorrectionModelInput(props: CorrectionModelInputProps) {
  const { key, model, onChange, onRemove } = props;

  return <Stack direction='row' spacing={2} key={key}>
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
          format: selection.target.value as ModelFormat,
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
          model: selection.target.value as CorrectionArch,
        });
      }}
    >
      <MenuItem value='codeformer'>Codeformer</MenuItem>
      <MenuItem value='gfpgan'>GFPGAN</MenuItem>
    </Select>
    <Button onClick={() => onRemove(model)}>Remove</Button>
  </Stack>;
}
