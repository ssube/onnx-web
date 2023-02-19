import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Casino } from '@mui/icons-material';
import { Button, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useQuery } from 'react-query';
import { useStore } from 'zustand';

import { BaseImgParams } from '../../client.js';
import { STALE_TIME } from '../../config.js';
import { ClientContext, ConfigContext, OnnxState, StateContext } from '../../state.js';
import { SCHEDULER_LABELS } from '../../strings.js';
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

  const client = mustExist(useContext(ClientContext));
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  return <Stack spacing={2}>
    <QueryList
      id='schedulers'
      labels={SCHEDULER_LABELS}
      name='Scheduler'
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
    <Stack direction='row' spacing={4}>
      <NumericField
        decimal
        label='CFG'
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
        label='Steps'
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
        label='Seed'
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
        New Seed
      </Button>
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
