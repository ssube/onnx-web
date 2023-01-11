import { mustExist } from '@apextoaster/js-utils';
import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useStore } from 'zustand';

import { ConfigParams } from '../config.js';
import { StateContext } from '../main.js';

const { useContext } = React;

export interface SettingsProps {
  config: ConfigParams;
}

export function Settings(_props: SettingsProps) {
  const state = useStore(mustExist(useContext(StateContext)));

  return <Stack spacing={2}>
    <Stack direction='row' spacing={2}>
      <Button onClick={() => state.resetTxt2Img()}>Reset Txt2Img</Button>
      <Button onClick={() => state.resetImg2Img()}>Reset Img2Img</Button>
      <Button onClick={() => state.resetInpaint()}>Reset Inpaint</Button>
      <Button disabled>Reset All</Button>
    </Stack>
    <TextField variant='outlined' label='Default Model' value={state.defaults.model} onChange={(event) => {
      state.setDefaults({
        model: event.target.value,
      });
    }} />
    <TextField variant='outlined' label='Default Platform' value={state.defaults.platform} onChange={(event) => {
      state.setDefaults({
        platform: event.target.value,
      });
    }} />
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
  </Stack>;
}
