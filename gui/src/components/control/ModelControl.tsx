import { mustExist } from '@apextoaster/js-utils';
import { Button, Stack } from '@mui/material';
import { useMutation, useQuery } from '@tanstack/react-query';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';

import { STALE_TIME } from '../../config.js';
import { ClientContext } from '../../state.js';
import { ModelParams } from '../../types/params.js';
import { QueryList } from '../input/QueryList.js';

export interface ModelControlProps {
  model: ModelParams;
  setModel(params: Partial<ModelParams>): void;
}

export function ModelControl(props: ModelControlProps) {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const { model, setModel } = props;

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

  return <Stack direction='row' spacing={2}>
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
