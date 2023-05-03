import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Casino } from '@mui/icons-material';
import { Button, Checkbox, FormControlLabel, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import { useStore } from 'zustand';

import { BaseImgParams } from '../../client/api.js';
import { STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state.js';
import { NumericField } from '../input/NumericField.js';
import { PromptInput } from '../input/PromptInput.js';
import { QueryList } from '../input/QueryList.js';

export interface ImageControlProps {
  selector: (state: OnnxState) => BaseImgParams;

  onChange?: (params: BaseImgParams) => void;
}

/**
 * Doesn't need to use state directly, the parent component knows which params to pass
 */
export function ImageControl(props: ImageControlProps) {
  const { params } = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  const controlState = useStore(state, props.selector);
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
        value={mustDefault(controlState.scheduler, '')}
        onChange={(value) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
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
        value={controlState.eta}
        onChange={(eta) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
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
        value={controlState.cfg}
        onChange={(cfg) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
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
        value={controlState.steps}
        onChange={(steps) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              steps,
            });
          }
        }}
      />
      <NumericField
        label={t('parameter.seed')}
        min={params.seed.min}
        max={params.seed.max}
        step={params.seed.step}
        value={controlState.seed}
        onChange={(seed) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              seed,
            });
          }
        }}
      />
      <Button
        variant='outlined'
        startIcon={<Casino />}
        onClick={() => {
          const seed = Math.floor(Math.random() * params.seed.max);
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              seed,
            });
          }
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
        value={controlState.batch}
        onChange={(batch) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              batch,
            });
          }
        }}
      />
      <NumericField
        label={t('parameter.tiles')}
        min={params.tiles.min}
        max={params.tiles.max}
        step={params.tiles.step}
        value={controlState.tiles}
        onChange={(tiles) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              tiles,
            });
          }
        }}
      />
      <NumericField
        decimal
        label={t('parameter.overlap')}
        min={params.overlap.min}
        max={params.overlap.max}
        step={params.overlap.step}
        value={controlState.overlap}
        onChange={(overlap) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...controlState,
              overlap,
            });
          }
        }}
      />
      <FormControlLabel
        label={t('parameter.tiledVAE')}
        control={<Checkbox
          checked={controlState.tiledVAE}
          value='check'
          onChange={(event) => {
            if (doesExist(props.onChange)) {
              props.onChange({
                ...controlState,
                tiledVAE: controlState.tiledVAE === false,
              });
            }
          }}
        />}
      />
    </Stack>
    <PromptInput
      prompt={controlState.prompt}
      negativePrompt={controlState.negativePrompt}
      onChange={(value) => {
        if (doesExist(props.onChange)) {
          props.onChange({
            ...controlState,
            ...value,
          });
        }
      }}
    />
  </Stack>;
}
