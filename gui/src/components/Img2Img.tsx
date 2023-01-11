import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation } from 'react-query';
import { useStore } from 'zustand';

import { equalResponse } from '../api/client.js';
import { ConfigParams, IMAGE_FILTER } from '../config.js';
import { ClientContext, StateContext } from '../main.js';
import { ImageCard } from './ImageCard.js';
import { ImageControl } from './ImageControl.js';
import { ImageInput } from './ImageInput.js';
import { MutationHistory } from './MutationHistory.js';
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
    return client.img2img({
      ...state.img2img,
      model,
      platform,
      source: mustExist(source), // TODO: show an error if this doesn't exist
    });
  }

  const client = mustExist(useContext(ClientContext));
  const upload = useMutation(uploadSource);
  const state = useStore(mustExist(useContext(StateContext)));

  const [source, setSource] = useState<File>();

  return <Box>
    <Stack spacing={2}>
      <ImageInput filter={IMAGE_FILTER} label='Source' onChange={setSource} />
      <ImageControl config={config} params={state.img2img} onChange={(newParams) => {
        state.setImg2Img(newParams);
      }} />
      <NumericField
        decimal
        label='Strength'
        min={config.strength.min}
        max={config.strength.max}
        step={config.strength.step}
        value={state.img2img.strength}
        onChange={(value) => {
          state.setImg2Img({
            strength: value,
          });
        }}
      />
      <Button onClick={() => upload.mutate()}>Generate</Button>
      <MutationHistory result={upload} limit={4} element={ImageCard}
        isEqual={equalResponse}
      />
    </Stack>
  </Box>;
}
