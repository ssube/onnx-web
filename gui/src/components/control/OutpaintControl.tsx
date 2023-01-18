import { mustExist } from '@apextoaster/js-utils';
import { ZoomOutMap } from '@mui/icons-material';
import { Stack, ToggleButton } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { ConfigContext, StateContext } from '../../state.js';
import { NumericField } from '../input/NumericField.js';

export function OutpaintControl() {
  const { params } = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  const outpaint = useStore(state, (s) => s.outpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setOutpaint = useStore(state, (s) => s.setOutpaint);

  return <Stack direction='row' spacing={4}>
    <ToggleButton
      color='primary'
      selected={outpaint.enabled}
      value='check'
      onChange={(event) => {
        setOutpaint({
          enabled: outpaint.enabled === false,
        });
      }}
    >
      <ZoomOutMap />
      Outpaint
    </ToggleButton>
    <NumericField
      label='Left'
      disabled={outpaint.enabled === false}
      min={0}
      max={params.width.max}
      step={params.width.step}
      value={outpaint.left}
      onChange={(left) => {
        setOutpaint({
          left,
        });
      }}
    />
    <NumericField
      label='Right'
      disabled={outpaint.enabled === false}
      min={0}
      max={params.width.max}
      step={params.width.step}
      value={outpaint.right}
      onChange={(right) => {
        setOutpaint({
          right,
        });
      }}
    />
    <NumericField
      label='Top'
      disabled={outpaint.enabled === false}
      min={0}
      max={params.height.max}
      step={params.height.step}
      value={outpaint.top}
      onChange={(top) => {
        setOutpaint({
          top,
        });
      }}
    />
    <NumericField
      label='Bottom'
      disabled={outpaint.enabled === false}
      min={0}
      max={params.height.max}
      step={params.height.step}
      value={outpaint.bottom}
      onChange={(bottom) => {
        setOutpaint({
          bottom,
        });
      }}
    />
  </Stack>;
}
