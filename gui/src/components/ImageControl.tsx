import { doesExist } from '@apextoaster/js-utils';
import { IconButton, Stack, TextField } from '@mui/material';
import { Casino } from '@mui/icons-material';
import * as React from 'react';

import { NumericField } from './NumericField.js';
import { BaseImgParams } from '../api/client.js';

export interface ImageControlProps {
  params: BaseImgParams;
  onChange?: (params: BaseImgParams) => void;
}

export function ImageControl(props: ImageControlProps) {
  const { params } = props;

  return <Stack spacing={2}>
    <Stack direction='row' spacing={4}>
      <NumericField
        label='CFG'
        min={0}
        max={30}
        step={1}
        value={params.cfg}
        onChange={(cfg) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              cfg,
            });
          }
        }}
      />
      <NumericField
        label='Steps'
        min={1}
        max={150}
        step={1}
        value={params.steps}
        onChange={(steps) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              steps,
            });
          }
        }}
      />
    </Stack>
    <Stack direction='row' spacing={4}>
      <NumericField
        label='Seed'
        min={-1}
        max={Number.MAX_SAFE_INTEGER}
        step={1}
        value={params.seed}
        onChange={(seed) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              seed,
            });
          }
        }}
      />
      <IconButton onClick={() => {
        const seed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
        if (doesExist(props.onChange)) {
          props.onChange({
            ...params,
            seed,
          });
        }
      }}>
        <Casino />
      </IconButton>
    </Stack>
    <TextField label='Prompt' variant='outlined' value={params.prompt} onChange={(event) => {
      if (doesExist(props.onChange)) {
        props.onChange({
          ...params,
          prompt: event.target.value,
        });
      }
    }} />
    <TextField label='Negative Prompt' variant='outlined' value={params.negativePrompt} onChange={(event) => {
      if (doesExist(props.onChange)) {
        props.onChange({
          ...params,
          negativePrompt: event.target.value,
        });
      }
    }} />
  </Stack>;
}
