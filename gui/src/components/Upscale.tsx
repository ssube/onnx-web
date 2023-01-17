import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER } from '../config.js';
import { ClientContext, ConfigContext, StateContext } from '../state.js';
import { ImageInput } from './ImageInput.js';
import { NumericField } from './NumericField.js';
import { UpscaleControl } from './UpscaleControl.js';

const { useContext } = React;

export function Upscale() {
  const config = mustExist(useContext(ConfigContext));

  async function uploadSource() {
    const { model, upscale } = state.getState();

    const output = await client.upscale(model, {
      ...params,
      source: mustExist(params.source), // TODO: show an error if this doesn't exist
    }, upscale);

    setLoading(output);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.upscaleTab);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setSource = useStore(state, (s) => s.setUpscaleTab);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);

  return <Box>
    <Stack spacing={2}>
      <ImageInput filter={IMAGE_FILTER} image={params.source} label='Source' onChange={(file) => {
        setSource({
          source: file,
        });
      }} />
      <NumericField
        decimal
        label='Strength'
        min={config.strength.min}
        max={config.strength.max}
        step={config.strength.step}
        value={params.strength}
        onChange={(value) => {
          setSource({
            strength: value,
          });
        }}
      />
      <UpscaleControl config={config} />
      <Button onClick={() => upload.mutate()}>Generate</Button>
    </Stack>
  </Box>;
}
