import { mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { ConfigContext, StateContext } from '../../state.js';
import { NumericField } from '../input/NumericField.js';

export function UpscaleControl() {
  const { params } = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  const highres = useStore(state, (s) => s.highres);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setHighres = useStore(state, (s) => s.setHighres);
  const { t } = useTranslation();

  return <Stack direction='row' spacing={4}>
    <FormControlLabel
      label={t('parameter.highres.label')}
      control={<Checkbox
        checked={highres.enabled}
        value='check'
        onChange={(event) => {
          setHighres({
            enabled: highres.enabled === false,
          });
        }}
      />}
    />
    <NumericField
      label={t('parameter.highres.steps')}
      decimal
      disabled={highres.enabled === false}
      min={params.denoise.min}
      max={params.denoise.max}
      step={params.denoise.step}
      value={highres.highresSteps}
      onChange={(steps) => {
        setHighres({
          highresSteps: steps,
        });
      }}
    />
    <NumericField
      label={t('parameter.highres.scale')}
      disabled={highres.enabled === false}
      min={params.scale.min}
      max={params.scale.max}
      step={params.scale.step}
      value={highres.highresScale}
      onChange={(scale) => {
        setHighres({
          highresScale: scale,
        });
      }}
    />
    <NumericField
      label={t('parameter.highres.strength')}
      disabled={highres.enabled === false}
      min={params.strength.min}
      max={params.strength.max}
      step={params.outscale.step}
      value={highres.highresStrength}
      onChange={(strength) => {
        setHighres({
          highresStrength: strength,
        });
      }}
    />
  </Stack>;
}
