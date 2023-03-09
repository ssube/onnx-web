import { mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControlLabel, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { ConfigContext, StateContext } from '../../state.js';
import { NumericField } from '../input/NumericField.js';

export function OutpaintControl() {
  const { params } = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  const outpaint = useStore(state, (s) => s.outpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setOutpaint = useStore(state, (s) => s.setOutpaint);
  const { t } = useTranslation();

  return <Stack direction='row' spacing={4}>
    <FormControlLabel
      label={t('parameter.outpaint.label')}
      control={<Checkbox
        checked={outpaint.enabled}
        value='check'
        onChange={(_event) => {
          setOutpaint({
            enabled: outpaint.enabled === false,
          });
        }}
      />}
    />
    <NumericField
      label={t('parameter.outpaint.left')}
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
      label={t('parameter.outpaint.right')}
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
      label={t('parameter.outpaint.top')}
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
      label={t('parameter.outpaint.bottom')}
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
