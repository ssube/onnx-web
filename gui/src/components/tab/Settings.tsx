import { mustExist } from '@apextoaster/js-utils';
import { Refresh } from '@mui/icons-material';
import { Alert, Button, Chip, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useContext, useState } from 'react';
import { useStore } from 'zustand';

import { getApiRoot } from '../../config.js';
import { ConfigContext, StateContext } from '../../state.js';
import { NumericField } from '../input/NumericField.js';

export function Settings() {
  const config = mustExist(useContext(ConfigContext));
  const state = useStore(mustExist(useContext(StateContext)));

  const [root, setRoot] = useState(getApiRoot(config));

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
      <TextField variant='outlined' label='API Server' value={root} onChange={(event) => {
        setRoot(event.target.value);
      }} />
      <Button variant='contained' startIcon={<Refresh />} onClick={() => {
        const query = new URLSearchParams(window.location.search);
        query.set('api', root);
        window.location.search = query.toString();
      }} />
      <Alert variant='outlined' severity='success'>
        {config.params.version}
      </Alert>
    </Stack>
    <Stack direction='row' spacing={2}>
      <Button onClick={() => state.resetTxt2Img()} color='warning'>Reset Txt2Img</Button>
      <Button onClick={() => state.resetImg2Img()} color='warning'>Reset Img2Img</Button>
      <Button onClick={() => state.resetInpaint()} color='warning'>Reset Inpaint</Button>
      <Button onClick={() => state.resetAll()} color='error'>Reset All</Button>
    </Stack>
  </Stack>;
}
