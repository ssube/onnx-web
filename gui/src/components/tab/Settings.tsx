import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Refresh } from '@mui/icons-material';
import { Alert, Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useContext, useState } from 'react';
import { useStore } from 'zustand';

import { getApiRoot } from '../../config.js';
import { ConfigContext, StateContext, STATE_KEY } from '../../state.js';
import { NumericField } from '../input/NumericField.js';

function removeBlobs(key: string, value: unknown): unknown {
  if (value instanceof Blob || value instanceof File) {
    return undefined;
  }

  if (Array.isArray(value)) {
    // check the first item, but return all of them
    if (doesExist(removeBlobs(key, value[0]))) {
      return value;
    }

    return [];
  }

  return value;
}

export function Settings() {
  const config = mustExist(useContext(ConfigContext));
  const state = useStore(mustExist(useContext(StateContext)));

  const [json, setJson] = useState(JSON.stringify(state, removeBlobs));
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
      }}>
        Connect
      </Button>
      <Alert variant='outlined' severity='success'>
        {config.params.version}
      </Alert>
    </Stack>
    <Stack direction='row' spacing={2}>
      <TextField variant='outlined' label='Client State' value={json} onChange={(event) => {
        setJson(event.target.value);
      }} />
      <Button variant='contained' startIcon={<Refresh />} onClick={() => {
        window.localStorage.setItem(STATE_KEY, json);
        window.location.reload();
      }}>
        Load
      </Button>
    </Stack>
    <Stack direction='row' spacing={2}>
      <Button onClick={() => state.resetTxt2Img()} color='warning'>Reset Txt2Img</Button>
      <Button onClick={() => state.resetImg2Img()} color='warning'>Reset Img2Img</Button>
      <Button onClick={() => state.resetInpaint()} color='warning'>Reset Inpaint</Button>
      <Button onClick={() => state.resetAll()} color='error'>Reset All</Button>
    </Stack>
  </Stack>;
}
