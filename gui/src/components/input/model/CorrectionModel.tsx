import { Stack, TextField } from '@mui/material';
import * as React from 'react';

import { CorrectionModel } from '../../../types';

export interface CorrectionModelInputProps {
  model: CorrectionModel;
}

export function CorrectionModelInput(props: CorrectionModelInputProps) {
  const { model } = props;

  return <Stack direction='row' spacing={2}>
    <TextField value={model.label} />
    <TextField value={model.source} />
  </Stack>;
}
