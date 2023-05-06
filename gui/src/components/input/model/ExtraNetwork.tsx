import { MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { ExtraNetwork } from '../../../types';

export interface ExtraNetworkInputProps {
  model: ExtraNetwork;
}

export function ExtraNetworkInput(props: ExtraNetworkInputProps) {
  const { model } = props;

  return <Stack direction='row' spacing={2}>
    <TextField value={model.label} label='Label' />
    <TextField value={model.source} label='Source' />
    <Select value={model.type} label='Type'>
      <MenuItem value='inversion'>Textual Inversion</MenuItem>
      <MenuItem value='lora'>LoRA or LyCORIS</MenuItem>
    </Select>
    <Select value={model.model} label='Model'>
      <MenuItem value='sd-scripts'>LoRA - sd-scripts</MenuItem>
      <MenuItem value='concept'>TI - concept</MenuItem>
      <MenuItem value='embeddings'>TI - embeddings</MenuItem>
    </Select>
  </Stack>;
}
