import { doesExist } from '@apextoaster/js-utils';
import { Casino } from '@mui/icons-material';
import { IconButton, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { BaseImgParams } from '../api/client.js';
import { CONFIG_DEFAULTS } from '../config.js';
import { NumericField } from './NumericField.js';

export interface ImageControlProps {
  params: BaseImgParams;
  onChange?: (params: BaseImgParams) => void;
}

export function ImageControl(props: ImageControlProps) {
  const { params } = props;

  return <Stack spacing={2}>
    <Stack direction='row' spacing={4}>
      <NumericField
        decimal
        label='CFG'
        min={CONFIG_DEFAULTS.cfg.min}
        max={CONFIG_DEFAULTS.cfg.max}
        step={CONFIG_DEFAULTS.cfg.step}
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
        min={CONFIG_DEFAULTS.steps.min}
        max={CONFIG_DEFAULTS.steps.max}
        step={CONFIG_DEFAULTS.steps.step}
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
      <NumericField
        label='Seed'
        min={CONFIG_DEFAULTS.seed.min}
        max={CONFIG_DEFAULTS.seed.max}
        step={CONFIG_DEFAULTS.seed.step}
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
        const seed = Math.floor(Math.random() * CONFIG_DEFAULTS.seed.max);
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
