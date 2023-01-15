import { mustExist } from '@apextoaster/js-utils';
import { Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { ConfigParams } from '../config.js';
import { StateContext } from '../state.js';
import { NumericField } from './NumericField.js';

export interface OutpaintControlProps {
  config: ConfigParams;
}

export function OutpaintControl(props: OutpaintControlProps) {
  const { config } = props;

  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.outpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setOutpaint = useStore(state, (s) => s.setOutpaint);

  return <Stack direction='row' spacing={4}>
    <NumericField
      label='Left'
      min={0}
      max={config.width.max}
      step={config.width.step}
      value={params.left}
      onChange={(left) => {
        setOutpaint({
          left,
        });
      }}
    />
    <NumericField
      label='Right'
      min={0}
      max={config.width.max}
      step={config.width.step}
      value={params.right}
      onChange={(right) => {
        setOutpaint({
          right,
        });
      }}
    />
    <NumericField
      label='Top'
      min={0}
      max={config.height.max}
      step={config.height.step}
      value={params.top}
      onChange={(top) => {
        setOutpaint({
          top,
        });
      }}
    />
    <NumericField
      label='Bottom'
      min={0}
      max={config.height.max}
      step={config.height.step}
      value={params.bottom}
      onChange={(bottom) => {
        setOutpaint({
          bottom,
        });
      }}
    />
  </Stack>;
}
