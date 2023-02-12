import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { IMAGE_FILTER } from '../../config.js';
import { ClientContext, StateContext } from '../../state.js';
import { UpscaleControl } from '../control/UpscaleControl.js';
import { ImageInput } from '../input/ImageInput.js';
import { MaskCanvas } from '../input/MaskCanvas.js';

export function Blend() {
  async function uploadSource() {
    const { model, blend, upscale } = state.getState();

    const output = await client.blend(model, {
      ...blend,
      mask: mustExist(blend.mask),
      sources: mustExist(blend.sources), // TODO: show an error if this doesn't exist
    }, upscale);

    setLoading(output);
  }

  const client = mustExist(useContext(ClientContext));
  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  const state = mustExist(useContext(StateContext));
  const blend = useStore(state, (s) => s.blend);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setBlend = useStore(state, (s) => s.setBlend);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.pushLoading);

  const sources = mustDefault(blend.sources, []);

  return <Box>
    <Stack spacing={2}>
      <ImageInput
        filter={IMAGE_FILTER}
        image={sources[0]}
        hideSelection={true}
        label='Source'
        onChange={(file) => {
          setBlend({
            sources: [file],
          });
        }}
      />
      <MaskCanvas
        source={sources[0]}
        mask={blend.mask}
        onSave={() => {
          // TODO
        }}
      />
      <UpscaleControl />
      <Button
        disabled={sources.length === 0}
        variant='contained'
        onClick={() => upload.mutate()}
      >Generate</Button>
    </Stack>
  </Box>;
}
