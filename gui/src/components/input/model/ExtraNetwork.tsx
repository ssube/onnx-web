import { MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { ExtraNetwork } from '../../../types.js';

export interface ExtraNetworkInputProps {
  model: ExtraNetwork;

  onChange: (model: ExtraNetwork) => void;
}

export function ExtraNetworkInput(props: ExtraNetworkInputProps) {
  const { model, onChange } = props;

  return <Stack direction='row' spacing={2}>
    <TextField
      label='Label'
      value={model.label}
      onChange={(event) => {
        onChange({
          ...model,
          label: event.target.value,
        });
      }}
    />
    <TextField
      label='Source'
      value={model.source}
      onChange={(event) => {
        onChange({
          ...model,
          source: event.target.value,
        });
      }}
    />
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
      <MenuItem value='bin'>bin</MenuItem>
    </Select>
    <Select value={model.type} label='Type' onChange={(selection) => {
      onChange({
        ...model,
        type: selection.target.value as 'lora',
      });
    }}>
      <MenuItem value='inversion'>Textual Inversion</MenuItem>
      <MenuItem value='lora'>LoRA or LyCORIS</MenuItem>
    </Select>
    <Select value={model.model} label='Model' onChange={(selection) => {
      onChange({
        ...model,
        model: selection.target.value as 'sd-scripts',
      });
    }}>
      <MenuItem value='sd-scripts'>LoRA - sd-scripts</MenuItem>
      <MenuItem value='concept'>TI - concept</MenuItem>
      <MenuItem value='embeddings'>TI - embeddings</MenuItem>
    </Select>
  </Stack>;
}
