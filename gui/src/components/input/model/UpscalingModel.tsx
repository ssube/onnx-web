import { Stack, TextField } from '@mui/material';
import * as React from 'react';

import { UpscalingModel } from '../../../types.js';

export interface UpscalingModelInputProps {
  model: UpscalingModel;
}

export function UpscalingModelInput(props: UpscalingModelInputProps) {
  const { model } = props;

  return <Stack direction='row' spacing={2}>
    <TextField value={model.label} />
    <TextField value={model.source} />
  </Stack>;
}
