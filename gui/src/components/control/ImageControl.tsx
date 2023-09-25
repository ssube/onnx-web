import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Casino } from '@mui/icons-material';
import { Button, Checkbox, FormControlLabel, Stack } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { omit } from 'lodash';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state.js';
import { BaseImgParams } from '../../types/params.js';
import { NumericField } from '../input/NumericField.js';
import { PromptInput } from '../input/PromptInput.js';
import { QueryList } from '../input/QueryList.js';

const { useMemo } = React;

type BaseParamsWithoutPrompt = Omit<BaseImgParams, 'prompt' | 'negativePrompt'>;

export interface ImageControlProps {
  onChange(params: Partial<BaseImgParams>): void;
  selector(state: OnnxState): BaseImgParams;
}

export function omitPrompt(selector: (state: OnnxState) => BaseImgParams): (state: OnnxState) => BaseParamsWithoutPrompt {
  return (state) => omit(selector(state), 'prompt', 'negativePrompt');
}

/**
 * Doesn't need to use state directly, the parent component knows which params to pass
 */
export function ImageControl(props: ImageControlProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { onChange, selector } = props;

  const selectOmitPrompt = useMemo(() => omitPrompt(selector), [selector]);
  const { params } = mustExist(useContext(ConfigContext));
  const store = mustExist(useContext(StateContext));
  const state = useStore(store, selectOmitPrompt, shallow);
  const { t } = useTranslation();

  const client = mustExist(useContext(ClientContext));
  const schedulers = useQuery(['schedulers'], async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  // max stride is the lesser of tile size and server's max stride
  const maxStride = Math.min(state.tiles, params.stride.max);

  return <Stack spacing={2}>
    <Stack direction='row' spacing={4}>
      <QueryList
        id='schedulers'
        labelKey='scheduler'
        name={t('parameter.scheduler')}
        query={{
          result: schedulers,
        }}
        value={mustDefault(state.scheduler, '')}
        onChange={(value) => {
          if (doesExist(onChange)) {
            onChange({
              ...state,
              scheduler: value,
            });
          }
        }}
      />
      <NumericField
        decimal
        label={t('parameter.eta')}
        min={params.eta.min}
        max={params.eta.max}
        step={params.eta.step}
        value={state.eta}
        onChange={(eta) => {
          if (doesExist(onChange)) {
            onChange({
              ...state,
              eta,
            });
          }
        }}
      />
      <NumericField
        decimal
        label={t('parameter.cfg')}
        min={params.cfg.min}
        max={params.cfg.max}
        step={params.cfg.step}
        value={state.cfg}
        onChange={(cfg) => {
          if (doesExist(onChange)) {
            onChange({
              ...state,
              cfg,
            });
          }
        }}
      />
      <NumericField
        label={t('parameter.steps')}
        min={params.steps.min}
        max={params.steps.max}
        step={params.steps.step}
        value={state.steps}
        onChange={(steps) => {
          onChange({
            ...state,
            steps,
          });
        }}
      />
      <NumericField
        label={t('parameter.seed')}
        min={params.seed.min}
        max={params.seed.max}
        step={params.seed.step}
        value={state.seed}
        onChange={(seed) => {
          onChange({
            ...state,
            seed,
          });
        }}
      />
      <Button
        variant='outlined'
        startIcon={<Casino />}
        onClick={() => {
          const seed = Math.floor(Math.random() * params.seed.max);
          props.onChange({
            ...state,
            seed,
          });
        }}
      >
        {t('parameter.newSeed')}
      </Button>
    </Stack>
    <Stack direction='row' spacing={4}>
      <NumericField
        label={t('parameter.batch')}
        min={params.batch.min}
        max={params.batch.max}
        step={params.batch.step}
        value={state.batch}
        onChange={(batch) => {
          props.onChange({
            ...state,
            batch,
          });
        }}
      />
      <NumericField
        label={t('parameter.tiles')}
        min={params.tiles.min}
        max={params.tiles.max}
        step={params.tiles.step}
        value={state.tiles}
        onChange={(tiles) => {
          props.onChange({
            ...state,
            tiles,
          });
        }}
      />
      <NumericField
        decimal
        label={t('parameter.overlap')}
        min={params.overlap.min}
        max={params.overlap.max}
        step={params.overlap.step}
        value={state.overlap}
        onChange={(overlap) => {
          props.onChange({
            ...state,
            overlap,
          });
        }}
      />
      <NumericField
        label={t('parameter.stride')}
        min={params.stride.min}
        max={maxStride}
        step={params.stride.step}
        value={state.stride}
        onChange={(stride) => {
          props.onChange({
            ...state,
            stride,
          });
        }}
      />
      <FormControlLabel
        label={t('parameter.tiledVAE')}
        control={<Checkbox
          checked={state.tiledVAE}
          value='check'
          onChange={(event) => {
            props.onChange({
              ...state,
              tiledVAE: state.tiledVAE === false,
            });
          }}
        />}
      />
    </Stack>
    <PromptInput
      selector={selector}
      onChange={(value) => {
        props.onChange({
          ...state,
          ...value,
        });
      }}
    />
  </Stack>;
}
