import { doesExist, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Casino } from '@mui/icons-material';
import { Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useQuery } from 'react-query';

import { BaseImgParams } from '../api/client.js';
import { ConfigParams, STALE_TIME } from '../config.js';
import { ClientContext } from '../main.js';
import { SCHEDULER_LABELS } from '../strings.js';
import { NumericField } from './NumericField.js';
import { QueryList } from './QueryList.js';

const { useContext } = React;

export interface ImageControlProps {
  config: ConfigParams;
  params: BaseImgParams;

  onChange?: (params: BaseImgParams) => void;
}

/**
 * doesn't need to use state, the parent component knows which params to pass
 */
export function ImageControl(props: ImageControlProps) {
  const { config, params } = props;

  const client = mustExist(useContext(ClientContext));
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  return <Stack spacing={2}>
    <QueryList
      id='schedulers'
      labels={SCHEDULER_LABELS}
      name='Scheduler'
      result={schedulers}
      value={mustDefault(params.scheduler, '')}
      onChange={(value) => {
        if (doesExist(props.onChange)) {
          props.onChange({
            ...params,
            scheduler: value,
          });
        }
      }}
    />
    <Stack direction='row' spacing={4}>
      <NumericField
        decimal
        label='CFG'
        min={config.cfg.min}
        max={config.cfg.max}
        step={config.cfg.step}
        value={params.cfg}
        onChange={(cfg) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              cfg,
            });
          }
        }}
      />
      <NumericField
        label='Steps'
        min={config.steps.min}
        max={config.steps.max}
        step={config.steps.step}
        value={params.steps}
        onChange={(steps) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              steps,
            });
          }
        }}
      />
      <NumericField
        label='Seed'
        min={config.seed.min}
        max={config.seed.max}
        step={config.seed.step}
        value={params.seed}
        onChange={(seed) => {
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              seed,
            });
          }
        }}
      />
      <Button
        variant='outlined'
        startIcon={<Casino />}
        onClick={() => {
          const seed = Math.floor(Math.random() * config.seed.max);
          if (doesExist(props.onChange)) {
            props.onChange({
              ...params,
              seed,
            });
          }
        }}
      >
        New Seed
      </Button>
    </Stack>
    <TextField label='Prompt' variant='outlined' value={params.prompt} onChange={(event) => {
      if (doesExist(props.onChange)) {
        props.onChange({
          ...params,
          prompt: event.target.value,
        });
      }
    }} />
    <TextField label='Negative Prompt' variant='outlined' value={params.negativePrompt} onChange={(event) => {
      if (doesExist(props.onChange)) {
        props.onChange({
          ...params,
          negativePrompt: event.target.value,
        });
      }
    }} />
  </Stack>;
}
