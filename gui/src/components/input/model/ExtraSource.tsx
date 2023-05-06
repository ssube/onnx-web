import * as React from 'react';
import { Stack, TextField } from '@mui/material';

import { ExtraSource } from '../../../types';

export interface ExtraSourceInputProps {
  model: ExtraSource;
}

export function ExtraSourceInput(props: ExtraSourceInputProps) {
  const { model } = props;

  return <Stack direction='row' spacing={2}>
    <TextField label='dest' value={model.dest} />
    <TextField label='source' value={model.source} />
  </Stack>;
}
