/* eslint-disable camelcase */
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
        label={t('parameter.unet_tile')}
        min={params.unet_tile.min}
        max={params.unet_tile.max}
        step={params.unet_tile.step}
        value={state.unet_tile}
        onChange={(unet_tile) => {
          props.onChange({
            ...state,
            unet_tile,
          });
        }}
      />
      <NumericField
        decimal
        label={t('parameter.unet_overlap')}
        min={params.unet_overlap.min}
        max={params.unet_overlap.max}
        step={params.unet_overlap.step}
        value={state.unet_overlap}
        onChange={(unet_overlap) => {
          props.onChange({
            ...state,
            unet_overlap,
          });
        }}
      />
      <FormControlLabel
        label={t('parameter.tiled_vae')}
        control={<Checkbox
          checked={state.tiled_vae}
          value='check'
          onChange={(event) => {
            props.onChange({
              ...state,
              tiled_vae: state.tiled_vae === false,
            });
          }}
        />}
      />
      <NumericField
        disabled={state.tiled_vae === false}
        label={t('parameter.vae_tile')}
        min={params.vae_tile.min}
        max={params.vae_tile.max}
        step={params.vae_tile.step}
        value={state.vae_tile}
        onChange={(vae_tile) => {
          props.onChange({
            ...state,
            vae_tile,
          });
        }}
      />
      <NumericField
        decimal
        disabled={state.tiled_vae === false}
        label={t('parameter.vae_overlap')}
        min={params.vae_overlap.min}
        max={params.vae_overlap.max}
        step={params.vae_overlap.step}
        value={state.vae_overlap}
        onChange={(vae_overlap) => {
          props.onChange({
            ...state,
            vae_overlap,
          });
        }}
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
