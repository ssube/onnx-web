import { mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControlLabel, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { ConfigContext, StateContext } from '../../state.js';
import { NumericField } from '../input/NumericField.js';

export function UpscaleControl() {
  const { params } = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  const upscale = useStore(state, (s) => s.upscale);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setUpscale = useStore(state, (s) => s.setUpscale);

  return <Stack direction='row' spacing={4}>
    <FormControlLabel
      label='Upscale'
      control={<Checkbox
        checked={upscale.enabled}
        value='check'
        onChange={(event) => {
          setUpscale({
            enabled: upscale.enabled === false,
          });
        }}
      />}
    />
    <NumericField
      label='Scale'
      disabled={upscale.enabled === false}
      min={params.scale.min}
      max={params.scale.max}
      step={params.scale.step}
      value={upscale.scale}
      onChange={(scale) => {
        setUpscale({
          scale,
        });
      }}
    />
    <NumericField
      label='Outscale'
      disabled={upscale.enabled === false}
      min={params.outscale.min}
      max={params.outscale.max}
      step={params.outscale.step}
      value={upscale.outscale}
      onChange={(outscale) => {
        setUpscale({
          outscale,
        });
      }}
    />
    <NumericField
      label='Denoise'
      decimal
      disabled={upscale.enabled === false}
      min={params.denoise.min}
      max={params.denoise.max}
      step={params.denoise.step}
      value={upscale.denoise}
      onChange={(denoise) => {
        setUpscale({
          denoise,
        });
      }}
    />
    <FormControlLabel
      label='Face Correction'
      control={<Checkbox
        disabled={upscale.enabled === false}
        checked={upscale.enabled && upscale.faces}
        value='check'
        onChange={(event) => {
          setUpscale({
            faces: upscale.faces === false,
          });
        }}
      />}
    />
    <NumericField
      label='Strength'
      decimal
      disabled={upscale.enabled === false || upscale.faces === false}
      min={params.faceStrength.min}
      max={params.faceStrength.max}
      step={params.faceStrength.step}
      value={upscale.faceStrength}
      onChange={(faceStrength) => {
        setUpscale({
          faceStrength,
        });
      }}
    />
  </Stack>;
}
