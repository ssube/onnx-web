import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Download, Refresh } from '@mui/icons-material';
import { Alert, Button, FormControlLabel, Stack, Switch, TextField, useMediaQuery } from '@mui/material';
import * as React from 'react';
import { useContext, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useStore } from 'zustand';

import { getApiRoot } from '../../config.js';
import { ConfigContext, StateContext, STATE_KEY } from '../../state/full.js';
import { getTheme } from '../utils.js';
import { NumericField } from '../input/NumericField.js';
import { downloadAsJson } from '../../utils.js';
import { STANDARD_SPACING } from '../../constants.js';

function removeBlobs(key: string, value: unknown): unknown {
  if (value instanceof Blob || value instanceof File) {
    return undefined;
  }

  if (Array.isArray(value)) {
    // check the first item, but return all of them
    if (doesExist(removeBlobs(key, value[0]))) {
      return value;
    }

    return [];
  }

  return value;
}

export function Settings() {
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const config = mustExist(useContext(ConfigContext));
  const state = useStore(mustExist(useContext(StateContext)));
  const theme = getTheme(state.theme, prefersDarkMode);

  const [json, setJson] = useState(JSON.stringify(state, removeBlobs));
  const [root, setRoot] = useState(getApiRoot(config));
  const { t } = useTranslation();

  return <Stack spacing={STANDARD_SPACING}>
    <NumericField
      label={t('setting.history.limit')}
      min={2}
      max={40}
      step={1}
      value={state.limit}
      onChange={(value) => state.setLimit(value)}
    />
    <NumericField
      label={t('setting.history.width')}
      min={2}
      max={6}
      step={1}
      value={state.historyWidth}
      onChange={(value) => state.setHistoryWidth(value)}
    />
    <Button variant='contained' onClick={() => state.setLayout(state.layout === 'horizontal' ? 'vertical' : 'horizontal')}>Toggle Layout</Button>
    <TextField variant='outlined' label={t('setting.prompt')} value={state.defaults.prompt} onChange={(event) => {
      state.setDefaults({
        prompt: event.target.value,
      });
    }} />
    <TextField variant='outlined' label={t('setting.scheduler')} value={state.defaults.scheduler} onChange={(event) => {
      state.setDefaults({
        scheduler: event.target.value,
      });
    }} />
    <Stack direction='row' spacing={STANDARD_SPACING}>
      <TextField variant='outlined' label={t('setting.server')} value={root} onChange={(event) => {
        setRoot(event.target.value);
      }} />
      <Button variant='contained' startIcon={<Refresh />} onClick={() => {
        const query = new URLSearchParams(window.location.search);
        query.set('api', root);
        window.location.search = query.toString();
      }}>
        {t('setting.connectServer')}
      </Button>
      <Alert variant='outlined' severity='success'>
        {config.params.version}
      </Alert>
    </Stack>
    <Stack direction='row' spacing={STANDARD_SPACING}>
      <TextField variant='outlined' label={t('setting.state.label')} value={json} onChange={(event) => {
        setJson(event.target.value);
      }} />
      <Button variant='contained' startIcon={<Refresh />} onClick={() => {
        window.localStorage.setItem(STATE_KEY, json);
        window.location.reload();
      }}>
        {t('setting.state.load')}
      </Button>
      <Button variant='contained' startIcon={<Download />} onClick={() => {
        downloadAsJson(state, 'state.json');
      }}>
        {t('setting.state.save')}
      </Button>
    </Stack>
    <FormControlLabel control={
      <Switch checked={theme === 'dark'}
        onClick={() => {
          if (theme === 'light') {
            state.setTheme('dark');
          } else {
            state.setTheme('light');
          }
        }}
      />
    } label={t('setting.darkMode')} />
    <Stack direction='row' spacing={STANDARD_SPACING}>
      <Button onClick={() => state.resetTxt2Img()} color='warning'>{t('setting.reset.txt2img')}</Button>
      <Button onClick={() => state.resetImg2Img()} color='warning'>{t('setting.reset.img2img')}</Button>
      <Button onClick={() => state.resetInpaint()} color='warning'>{t('setting.reset.inpaint')}</Button>
      <Button onClick={() => state.resetAll()} color='error'>{t('setting.reset.all')}</Button>
    </Stack>
  </Stack>;
}
