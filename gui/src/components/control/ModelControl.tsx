import { mustExist } from '@apextoaster/js-utils';
import { Checkbox, FormControlLabel, Stack } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from 'react-query';
import { useHash } from 'react-use/lib/useHash';
import { useStore } from 'zustand';

import { STALE_TIME } from '../../config.js';
import { ClientContext, StateContext } from '../../state.js';
import { QueryList } from '../input/QueryList.js';
import { QueryMenu } from '../input/QueryMenu.js';
import { getTab } from '../utils.js';

export function ModelControl() {
  const client = mustExist(useContext(ClientContext));
  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.model);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setModel = useStore(state, (s) => s.setModel);
  const { t } = useTranslation();

  const models = useQuery('models', async () => client.models(), {
    staleTime: STALE_TIME,
  });
  const platforms = useQuery('platforms', async () => client.platforms(), {
    staleTime: STALE_TIME,
  });

  const [hash, _setHash] = useHash();

  function addToken(type: string, name: string, weight = 1.0) {
    const tab = getTab(hash);
    const current = state.getState();


    switch (tab) {
      case 'txt2img': {
        const { prompt } = current.txt2img;
        current.setTxt2Img({
          prompt: `<${type}:${name}:1.0> ${prompt}`,
        });
        break;
      }
      case 'img2img': {
        const { prompt } = current.img2img;
        current.setImg2Img({
          prompt: `<${type}:${name}:1.0> ${prompt}`,
        });
        break;
      }
      default:
        // not supported yet
    }
  }

  return <Stack direction='column' spacing={2}>
    <Stack direction='row' spacing={2}>
      <QueryList
        id='platforms'
        labelKey='platform'
        name={t('parameter.platform')}
        query={{
          result: platforms,
        }}
        value={params.platform}
        onChange={(platform) => {
          setModel({
            platform,
          });
        }}
      />
      <QueryList
        id='diffusion'
        labelKey='model'
        name={t('modelType.diffusion')}
        query={{
          result: models,
          selector: (result) => result.diffusion,
        }}
        value={params.model}
        onChange={(model) => {
          setModel({
            model,
          });
        }}
      />
      <QueryList
        id='upscaling'
        labelKey='model'
        name={t('modelType.upscaling')}
        query={{
          result: models,
          selector: (result) => result.upscaling,
        }}
        value={params.upscaling}
        onChange={(upscaling) => {
          setModel({
            upscaling,
          });
        }}
      />
      <QueryList
        id='correction'
        labelKey='model'
        name={t('modelType.correction')}
        query={{
          result: models,
          selector: (result) => result.correction,
        }}
        value={params.correction}
        onChange={(correction) => {
          setModel({
            correction,
          });
        }}
      />
    </Stack>
    <Stack direction='row' spacing={2}>
      <FormControlLabel
        label={t('parameter.lpw')}
        control={<Checkbox
          checked={params.lpw}
          value='check'
          onChange={(event) => {
            setModel({
              lpw: params.lpw === false,
            });
          }}
        />}
      />
      <QueryMenu
        id='inversion'
        labelKey='model.inversion'
        name={t('modelType.inversion')}
        query={{
          result: models,
          selector: (result) => result.networks.filter((network) => network.type === 'inversion').map((network) => network.name),
        }}
        onSelect={(name) => {
          addToken('inversion', name);
        }}
      />
      <QueryMenu
        id='lora'
        labelKey='model.lora'
        name={t('modelType.lora')}
        query={{
          result: models,
          selector: (result) => result.networks.filter((network) => network.type === 'lora').map((network) => network.name),
        }}
        onSelect={(name) => {
          addToken('lora', name);
        }}
      />
    </Stack>
  </Stack>;
}
