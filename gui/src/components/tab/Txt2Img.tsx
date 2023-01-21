import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { ClientContext, ConfigContext, StateContext } from '../../state.js';
import { ImageControl } from '../control/ImageControl.js';
import { NumericField } from '../input/NumericField.js';
import { UpscaleControl } from '../control/UpscaleControl.js';

const { useContext } = React;

export function Txt2Img() {
  const { params } = mustExist(useContext(ConfigContext));

  async function generateImage() {
    const { model, txt2img, upscale } = state.getState();
    const output = await client.txt2img(model, txt2img, upscale);

    setLoading(output);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const generate = useMutation(generateImage, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  const state = mustExist(useContext(StateContext));
  const height = useStore(state, (s) => s.txt2img.height);
  const width = useStore(state, (s) => s.txt2img.width);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setTxt2Img = useStore(state, (s) => s.setTxt2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);

  return <Box>
    <Stack spacing={2}>
      <ImageControl selector={(s) => s.txt2img} onChange={setTxt2Img} />
      <Stack direction='row' spacing={4}>
        <NumericField
          label='Width'
          min={params.width.min}
          max={params.width.max}
          step={params.width.step}
          value={width}
          onChange={(value) => {
            setTxt2Img({
              width: value,
            });
          }}
        />
        <NumericField
          label='Height'
          min={params.height.min}
          max={params.height.max}
          step={params.height.step}
          value={height}
          onChange={(value) => {
            setTxt2Img({
              height: value,
            });
          }}
        />
      </Stack>
      <UpscaleControl />
      <Button
        variant='outlined'
        onClick={() => generate.mutate()}
      >Generate</Button>
    </Stack>
  </Box>;
}
