import { mustExist } from '@apextoaster/js-utils';
import { Button, Stack } from '@mui/material';
import { useMutation, useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';

import { STALE_TIME, STANDARD_SPACING } from '../../constants.js';
import { ClientContext } from '../../state/full.js';
import { JobType } from '../../types/api-v2.js';
import { ModelParams } from '../../types/params.js';
import { QueryList } from '../input/QueryList.js';

export interface ModelControlProps {
  model: ModelParams;
  setModel(params: Partial<ModelParams>): void;
  tab: JobType;
}

export function ModelControl(props: ModelControlProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { model, setModel, tab } = props;

  const client = mustExist(useContext(ClientContext));
  const { t } = useTranslation();

  const restart = useMutation(['restart'], async () => client.restart());
  const models = useQuery(['models'], async () => client.models(), {
    staleTime: STALE_TIME,
  });
  const pipelines = useQuery(['pipelines'], async () => client.pipelines(), {
    staleTime: STALE_TIME,
  });
  const platforms = useQuery(['platforms'], async () => client.platforms(), {
    staleTime: STALE_TIME,
  });

  return <Stack direction='row' spacing={STANDARD_SPACING}>
    <QueryList
      id='platforms'
      labelKey='platform'
      name={t('parameter.platform')}
      query={{
        result: platforms,
      }}
      value={model.platform}
      onChange={(platform) => {
        setModel({
          platform,
        });
      }}
    />
    <QueryList
      id='pipeline'
      labelKey='pipeline'
      name={t('parameter.pipeline')}
      query={{
        result: pipelines,
        selector: (result) => filterValidPipelines(result, tab),
      }}
      value={model.pipeline}
      onChange={(pipeline) => {
        setModel({
          pipeline,
        });
      }}
    />
    <QueryList
      id='diffusion'
      labelKey='model'
      name={t('modelType.diffusion', { count: 1 })}
      query={{
        result: models,
        selector: (result) => result.diffusion,
      }}
      value={model.model}
      onChange={(newModel) => {
        setModel({
          model: newModel,
        });
      }}
    />
    <QueryList
      id='upscaling'
      labelKey='model'
      name={t('modelType.upscaling', { count: 1 })}
      query={{
        result: models,
        selector: (result) => result.upscaling,
      }}
      value={model.upscaling}
      onChange={(upscaling) => {
        setModel({
          upscaling,
        });
      }}
    />
    <QueryList
      id='correction'
      labelKey='model'
      name={t('modelType.correction', { count: 1 })}
      query={{
        result: models,
        selector: (result) => result.correction,
      }}
      value={model.correction}
      onChange={(correction) => {
        setModel({
          correction,
        });
      }}
    />
    <Button
      variant='outlined'
      onClick={() => restart.mutate()}
    >{t('admin.restart')}</Button>
  </Stack>;
}

// plugin pipelines will show up on all tabs for now
export const PIPELINE_TABS: Record<string, Array<JobType>> = {
  'txt2img': [JobType.TXT2IMG],
  'txt2img-sdxl': [JobType.TXT2IMG],
  'panorama': [JobType.TXT2IMG, JobType.IMG2IMG],
  'panorama-sdxl': [JobType.TXT2IMG, JobType.IMG2IMG],
  'lpw': [JobType.TXT2IMG, JobType.IMG2IMG, JobType.INPAINT],
  'img2img': [JobType.IMG2IMG],
  'img2img-sdxl': [JobType.IMG2IMG],
  'controlnet': [JobType.IMG2IMG],
  'pix2pix': [JobType.IMG2IMG],
  'inpaint': [JobType.INPAINT],
  'upscale': [JobType.UPSCALE],
};

export const DEFAULT_PIPELINE: Record<JobType, string> = {
  [JobType.TXT2IMG]: 'txt2img',
  [JobType.IMG2IMG]: 'img2img',
  [JobType.INPAINT]: 'inpaint',
  [JobType.UPSCALE]: 'upscale',
  [JobType.BLEND]: '',
  [JobType.CHAIN]: '',
};

export const FIRST_A = -1;
export const FIRST_B = +1;

export function filterValidPipelines(pipelines: Array<string>, tab: JobType): Array<string> {
  const defaultPipeline = DEFAULT_PIPELINE[tab];
  return pipelines.filter((pipeline) => PIPELINE_TABS[pipeline].includes(tab)).sort((a, b) => {
    // put validPipelines.default first
    if (a === defaultPipeline) {
      return FIRST_A;
    }

    if (b === defaultPipeline) {
      return FIRST_B;
    }

    return a.localeCompare(b);
  });
}
