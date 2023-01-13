import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation } from 'react-query';
import { useStore } from 'zustand';

import { ConfigParams } from '../config.js';
import { ClientContext, StateContext } from '../state.js';
import { ImageControl } from './ImageControl.js';
import { NumericField } from './NumericField.js';

const { useContext } = React;

export interface Txt2ImgProps {
  config: ConfigParams;

  model: string;
  platform: string;
}

export function Txt2Img(props: Txt2ImgProps) {
  const { config, model, platform } = props;

  async function generateImage() {
    setLoading(true);

    const output = await client.txt2img({
      ...params,
      model,
      platform,
    });

    pushHistory(output);
    setLoading(false);
  }

  const client = mustExist(useContext(ClientContext));
  const generate = useMutation(generateImage);

  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.txt2img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setTxt2Img = useStore(state, (s) => s.setTxt2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const pushHistory = useStore(state, (s) => s.pushHistory);

  return <Box>
    <Stack spacing={2}>
      <ImageControl config={config} params={params} onChange={(newParams) => {
        setTxt2Img(newParams);
      }} />
      <Stack direction='row' spacing={4}>
        <NumericField
          label='Width'
          min={config.width.min}
          max={config.width.max}
          step={config.width.step}
          value={params.width}
          onChange={(value) => {
            setTxt2Img({
              width: value,
            });
          }}
        />
        <NumericField
          label='Height'
          min={config.height.min}
          max={config.height.max}
          step={config.height.step}
          value={params.height}
          onChange={(value) => {
            setTxt2Img({
              height: value,
            });
          }}
        />
      </Stack>
      <Button onClick={() => generate.mutate()}>Generate</Button>
    </Stack>
  </Box>;
}
