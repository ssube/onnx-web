import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation } from 'react-query';
import { useStore } from 'zustand';

import { BaseImgParams, equalResponse, paramsFromConfig } from '../api/client.js';
import { ConfigParams } from '../config.js';
import { ClientContext, StateContext } from '../main.js';
import { ImageCard } from './ImageCard.js';
import { ImageControl } from './ImageControl.js';
import { MutationHistory } from './MutationHistory.js';
import { NumericField } from './NumericField.js';

const { useContext, useState } = React;

export interface Txt2ImgProps {
  config: ConfigParams;

  model: string;
  platform: string;
}

export function Txt2Img(props: Txt2ImgProps) {
  const { config, model, platform } = props;

  async function generateImage() {
    return client.txt2img({
      ...state.txt2img,
      model,
      platform,
    });
  }

  const client = mustExist(useContext(ClientContext));
  const generate = useMutation(generateImage);
  const state = useStore(mustExist(useContext(StateContext)));

  return <Box>
    <Stack spacing={2}>
      <ImageControl config={config} params={state.txt2img} onChange={(newParams) => {
        state.setTxt2Img(newParams);
      }} />
      <Stack direction='row' spacing={4}>
        <NumericField
          label='Width'
          min={config.width.min}
          max={config.width.max}
          step={config.width.step}
          value={state.txt2img.width}
          onChange={(value) => {
            state.setTxt2Img({
              width: value,
            });
          }}
        />
        <NumericField
          label='Height'
          min={config.height.min}
          max={config.height.max}
          step={config.height.step}
          value={state.txt2img.height}
          onChange={(value) => {
            state.setTxt2Img({
              height: value,
            });
          }}
        />
      </Stack>
      <Button onClick={() => generate.mutate()}>Generate</Button>
      <MutationHistory
        element={ImageCard}
        limit={4}
        isEqual={equalResponse}
        result={generate}
      />
    </Stack>
  </Box>;
}
