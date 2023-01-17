import { mustExist } from '@apextoaster/js-utils';
import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { StateContext } from '../state.js';
import { NumericField } from './NumericField.js';

export function Settings() {
  const state = useStore(mustExist(useContext(StateContext)));

  return <Stack spacing={2}>
    <NumericField
      label='Image History'
      min={2}
      max={20}
      step={1}
      value={state.limit}
      onChange={(value) => state.setLimit(value)}
    />
    <TextField variant='outlined' label='Default Prompt' value={state.defaults.prompt} onChange={(event) => {
      state.setDefaults({
        prompt: event.target.value,
      });
    }} />
    <TextField variant='outlined' label='Default Scheduler' value={state.defaults.scheduler} onChange={(event) => {
      state.setDefaults({
        scheduler: event.target.value,
      });
    }} />
    <Stack direction='row' spacing={2}>
      <Button onClick={() => state.resetTxt2Img()}>Reset Txt2Img</Button>
      <Button onClick={() => state.resetImg2Img()}>Reset Img2Img</Button>
      <Button onClick={() => state.resetInpaint()}>Reset Inpaint</Button>
      <Button disabled>Reset All</Button>
    </Stack>
  </Stack>;
}
