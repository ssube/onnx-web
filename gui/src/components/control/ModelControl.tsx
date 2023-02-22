import { mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControlLabel, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useQuery } from 'react-query';
import { useStore } from 'zustand';

import { STALE_TIME } from '../../config.js';
import { ClientContext, StateContext } from '../../state.js';
import { MODEL_LABELS, PLATFORM_LABELS } from '../../strings.js';
import { QueryList } from '../input/QueryList.js';

export function ModelControl() {
  const client = mustExist(useContext(ClientContext));
  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.model);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setModel = useStore(state, (s) => s.setModel);

  const models = useQuery('models', async () => client.models(), {
    staleTime: STALE_TIME,
  });
  const platforms = useQuery('platforms', async () => client.platforms(), {
    staleTime: STALE_TIME,
  });

  return <Stack direction='row' spacing={2}>
    <QueryList
      id='platforms'
      labels={PLATFORM_LABELS}
      name='Platform'
      query={{
        result: platforms,
      }}
      value={params.platform}
      onChange={(platform) => {
        setModel({
          platform,
        });
      }}
    />
    <QueryList
      id='diffusion'
      labels={MODEL_LABELS}
      name='Diffusion Model'
      query={{
        result: models,
        selector: (result) => result.diffusion,
      }}
      value={params.model}
      onChange={(model) => {
        setModel({
          model,
        });
      }}
    />
    <QueryList
      id='inversion'
      labels={MODEL_LABELS}
      name='Textual Inversion'
      query={{
        result: models,
        selector: (result) => result.inversion,
      }}
      value={params.inversion}
      onChange={(inversion) => {
        setModel({
          inversion,
        });
      }}
    />
    <QueryList
      id='upscaling'
      labels={MODEL_LABELS}
      name='Upscaling Model'
      query={{
        result: models,
        selector: (result) => result.upscaling,
      }}
      value={params.upscaling}
      onChange={(upscaling) => {
        setModel({
          upscaling,
        });
      }}
    />
    <QueryList
      id='correction'
      labels={MODEL_LABELS}
      name='Correction Model'
      query={{
        result: models,
        selector: (result) => result.correction,
      }}
      value={params.correction}
      onChange={(correction) => {
        setModel({
          correction,
        });
      }}
    />
    <FormControlLabel
      label='Long Prompt Weighting'
      control={<Checkbox
        checked={params.lpw}
        value='check'
        onChange={(event) => {
          setModel({
            lpw: params.lpw === false,
          });
        }}
      />}
    />
  </Stack>;
}
