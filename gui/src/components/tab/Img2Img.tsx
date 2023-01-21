import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER } from '../../config.js';
import { ClientContext, ConfigContext, StateContext } from '../../state.js';
import { ImageControl } from '../control/ImageControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { NumericField } from '../input/NumericField.js';
import { UpscaleControl } from '../control/UpscaleControl.js';

const { useContext } = React;

export function Img2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function uploadSource() {
    const { model, img2img, upscale } = state.getState();

    const output = await client.img2img(model, {
      ...img2img,
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
      <ImageControl selector={(s) => s.img2img} onChange={setImg2Img} />
      <NumericField
        decimal
        label='Strength'
        min={params.strength.min}
        max={params.strength.max}
        step={params.strength.step}
        value={strength}
        onChange={(value) => {
          setImg2Img({
            strength: value,
          });
        }}
      />
      <UpscaleControl />
      <Button
        disabled={doesExist(source) === false}
        variant='contained'
        onClick={() => upload.mutate()}
      >Generate</Button>
    </Stack>
  </Box>;
}
