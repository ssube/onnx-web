import { mustExist } from '@apextoaster/js-utils';
import { Check, FaceRetouchingNatural, ZoomIn } from '@mui/icons-material';
import { Stack, ToggleButton } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { ConfigParams } from '../config.js';
import { StateContext } from '../state.js';
import { NumericField } from './NumericField.js';

export interface UpscaleControlProps {
  config: ConfigParams;
}

export function UpscaleControl(props: UpscaleControlProps) {
  const { config } = props;

  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.upscale);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setUpscale = useStore(state, (s) => s.setUpscale);

  return <Stack direction='row' spacing={4}>
    <ToggleButton
      color='primary'
      selected={params.enabled}
      value='check'
      onChange={(event) => {
        setUpscale({
          enabled: params.enabled === false,
        });
      }}
    >
      <ZoomIn />
      Upscale
    </ToggleButton>
    <NumericField
      label='Scale'
      disabled={params.enabled === false}
      min={config.scale.min}
      max={config.scale.max}
      step={config.scale.step}
      value={params.scale}
      onChange={(scale) => {
        setUpscale({
          scale,
        });
      }}
    />
    <NumericField
      label='Outscale'
      disabled={params.enabled === false}
      min={config.outscale.min}
      max={config.outscale.max}
      step={config.outscale.step}
      value={params.outscale}
      onChange={(outscale) => {
        setUpscale({
          outscale,
        });
      }}
    />
    <NumericField
      label='Denoise'
      decimal
      disabled={params.enabled === false}
      min={config.denoise.min}
      max={config.denoise.max}
      step={config.denoise.step}
      value={params.denoise}
      onChange={(denoise) => {
        setUpscale({
          denoise,
        });
      }}
    />
    <ToggleButton
      color='primary'
      disabled={params.enabled === false}
      selected={params.enabled && params.faces}
      value='check'
      onChange={(event) => {
        setUpscale({
          faces: params.faces === false,
        });
      }}
    >
      <FaceRetouchingNatural />
      Face Correction
    </ToggleButton>
    <NumericField
      label='Strength'
      decimal
      disabled={params.enabled === false && params.faces === false}
      min={config.faceStrength.min}
      max={config.faceStrength.max}
      step={config.faceStrength.step}
      value={params.faceStrength}
      onChange={(faceStrength) => {
        setUpscale({
          faceStrength,
        });
      }}
    />
  </Stack>;
}
