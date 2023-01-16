import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { ConfigParams, IMAGE_FILTER, STALE_TIME } from '../config.js';
import { ClientContext, StateContext } from '../state.js';
import { MASK_LABELS, NOISE_LABELS } from '../strings.js';
import { ImageControl } from './ImageControl.js';
import { ImageInput } from './ImageInput.js';
import { MaskCanvas } from './MaskCanvas.js';
import { OutpaintControl } from './OutpaintControl.js';
import { QueryList } from './QueryList.js';
import { UpscaleControl } from './UpscaleControl.js';

const { useContext } = React;

export interface InpaintProps {
  config: ConfigParams;

  model: string;
  platform: string;
}

export function Inpaint(props: InpaintProps) {
  const { config, model, platform } = props;
  const client = mustExist(useContext(ClientContext));
  const masks = useQuery('masks', async () => client.masks(), {
    staleTime: STALE_TIME,
  });
  const noises = useQuery('noises', async () => client.noises(), {
    staleTime: STALE_TIME,
  });

  async function uploadSource(): Promise<void> {
    const outpaint = state.getState().outpaint; // TODO: seems shady

    if (outpaint.enabled) {
      const output = await client.outpaint({
        ...params,
        ...outpaint,
        model,
        platform,
        mask: mustExist(params.mask),
        source: mustExist(params.source),
      });

      setLoading(output);
    } else {
      const output = await client.inpaint({
        ...params,
        model,
        platform,
        mask: mustExist(params.mask),
        source: mustExist(params.source),
      });

      setLoading(output);
    }
  }

  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.inpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setInpaint = useStore(state, (s) => s.setInpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);

  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });

  return <Box>
    <Stack spacing={2}>
      <ImageInput
        filter={IMAGE_FILTER}
        image={params.source}
        label='Source'
        onChange={(file) => {
          setInpaint({
            source: file,
          });
        }}
      />
      <ImageInput
        filter={IMAGE_FILTER}
        image={params.mask}
        label='Mask'
        onChange={(file) => {
          setInpaint({
            mask: file,
          });
        }}
        renderImage={(image) =>
          <MaskCanvas
            config={config}
            base={params.source}
            source={image}
            onSave={(mask) => {
              setInpaint({
                mask,
              });
            }}
          />
        }
      />
      <ImageControl
        config={config}
        params={params}
        onChange={(newParams) => {
          setInpaint(newParams);
        }}
      />
      <Stack direction='row' spacing={2}>
        <QueryList
          id='masks'
          labels={MASK_LABELS}
          name='Mask Filter'
          result={masks}
          value={params.filter}
          onChange={(filter) => {
            setInpaint({
              filter,
            });
          }}
        />
        <QueryList
          id='noises'
          labels={NOISE_LABELS}
          name='Noise Source'
          result={noises}
          value={params.noise}
          onChange={(noise) => {
            setInpaint({
              noise,
            });
          }}
        />
      </Stack>
      <OutpaintControl config={config} />
      <UpscaleControl config={config} />
      <Button onClick={() => upload.mutate()}>Generate</Button>
    </Stack>
  </Box>;
}
