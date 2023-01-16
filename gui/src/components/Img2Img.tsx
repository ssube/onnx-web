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
    const { img2img, upscale } = state.getState();

    const output = await client.img2img({
      ...img2img,
      model,
      platform,
      source: mustExist(img2img.source), // TODO: show an error if this doesn't exist
    }, upscale);

    setLoading(output);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  const state = mustExist(useContext(StateContext));
  const source = useStore(state, (s) => s.img2img.source);
  const strength = useStore(state, (s) => s.img2img.strength);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setImg2Img = useStore(state, (s) => s.setImg2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);

  return <Box>
    <Stack spacing={2}>
      <ImageInput filter={IMAGE_FILTER} image={source} label='Source' onChange={(file) => {
        setImg2Img({
          source: file,
        });
      }} />
      <ImageControl config={config} selector={(s) => s.img2img} onChange={setImg2Img} />
      <NumericField
        decimal
        label='Strength'
        min={config.strength.min}
        max={config.strength.max}
        step={config.strength.step}
        value={strength}
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
