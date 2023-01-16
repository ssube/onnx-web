import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { ConfigParams, IMAGE_FILTER } from '../config.js';
import { ClientContext, StateContext } from '../state.js';
import { ImageControl } from './ImageControl.js';
import { ImageInput } from './ImageInput.js';
import { NumericField } from './NumericField.js';
import { UpscaleControl } from './UpscaleControl.js';

const { useContext } = React;

export interface Img2ImgProps {
  config: ConfigParams;

  model: string;
  platform: string;
}

export function Img2Img(props: Img2ImgProps) {
  const { config, model, platform } = props;

  async function uploadSource() {
    const upscale = state.getState().upscale;

    const output = await client.img2img({
      ...params,
      model,
      platform,
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
  const params = useStore(state, (s) => s.img2img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setImg2Img = useStore(state, (s) => s.setImg2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);

  return <Box>
    <Stack spacing={2}>
      <ImageInput filter={IMAGE_FILTER} image={params.source} label='Source' onChange={(file) => {
        setImg2Img({
          source: file,
        });
      }} />
      <ImageControl config={config} params={params} onChange={(newParams) => {
        setImg2Img(newParams);
      }} />
      <NumericField
        decimal
        label='Strength'
        min={config.strength.min}
        max={config.strength.max}
        step={config.strength.step}
        value={params.strength}
        onChange={(value) => {
          setImg2Img({
            strength: value,
          });
        }}
      />
      <UpscaleControl config={config} />
      <Button onClick={() => upload.mutate()}>Generate</Button>
    </Stack>
  </Box>;
}
