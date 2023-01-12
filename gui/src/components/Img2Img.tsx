import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation } from 'react-query';
import { useStore } from 'zustand';

import { ConfigParams, IMAGE_FILTER } from '../config.js';
import { ClientContext, StateContext } from '../main.js';
import { ImageControl } from './ImageControl.js';
import { ImageInput } from './ImageInput.js';
import { NumericField } from './NumericField.js';

const { useContext, useState } = React;

export interface Img2ImgProps {
  config: ConfigParams;

  model: string;
  platform: string;
}

export function Img2Img(props: Img2ImgProps) {
  const { config, model, platform } = props;

  async function uploadSource() {
    setLoading(true);

    const output = await client.img2img({
      ...params,
      model,
      platform,
      source: mustExist(source), // TODO: show an error if this doesn't exist
    });

    pushHistory(output);
    setLoading(false);
  }

  const client = mustExist(useContext(ClientContext));
  const upload = useMutation(uploadSource);

  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.img2img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setImg2Img = useStore(state, (s) => s.setImg2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);

  const [source, setSource] = useState<File>();

  return <Box>
    <Stack spacing={2}>
      <ImageInput filter={IMAGE_FILTER} label='Source' onChange={setSource} />
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
      <Button onClick={() => upload.mutate()}>Generate</Button>
    </Stack>
  </Box>;
}
