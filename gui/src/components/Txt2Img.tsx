import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useMutation, useQuery } from 'react-query';

import { ApiClient } from '../api/client.js';
import { ImageControl, ImageParams } from './ImageControl.js';

const { useState } = React;

export const STALE_TIME = 3_000;

// TODO: set up i18next
const PLATFORM_NAMES: Record<string, string> = {
  amd: 'AMD GPU',
  cpu: 'CPU',
};

const SCHEDULER_NAMES: Record<string, string> = {
  'ddim': 'DDIM',
  'ddpm': 'DDPM',
  'dpm-multi': 'DPM Multistep',
  'euler': 'Euler',
  'euler-a': 'Euler Ancestral',
  'lms-discrete': 'LMS Discrete',
  'pndm': 'PNDM',
};

export interface Txt2ImgProps {
  client: ApiClient;
  model: string;
}

export function Txt2Img(props: Txt2ImgProps) {
  const { client } = props;

  async function generateImage() {
    return client.txt2img({
      ...params,
      model: props.model,
      prompt,
      scheduler,
    });
  }

  const generate = useMutation(generateImage);
  const platforms = useQuery('platforms', async () => client.platforms(), {
    staleTime: STALE_TIME,
  });
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  const [prompt, setPrompt] = useState('an astronaut eating a hamburger');
  const [params, setParams] = useState<ImageParams>({
    cfg: 6,
    steps: 25,
    width: 512,
    height: 512,
  });
  const [scheduler, setScheduler] = useState('euler-a');
  const [platform, setPlatform] = useState('cpu');

  function renderImage() {
    switch (generate.status) {
      case 'error':
        if (generate.error instanceof Error) {
          return <div>{generate.error.message}</div>;
        } else {
          return <div>Unknown error generating image.</div>;
        }
      case 'loading':
        return <div>Generating...</div>;
      case 'success':
        return <img src={generate.data.output} />;
      default:
        return <div>No result. Press Generate.</div>;
    }
  }

  function renderSchedulers() {
    switch (schedulers.status) {
      case 'error':
        return <MenuItem value='error'>Error</MenuItem>;
      case 'loading':
        return <MenuItem value='loading'>Loading</MenuItem>;
      case 'success':
        return mustExist(schedulers.data).map((name) => <MenuItem key={name} value={name}>{SCHEDULER_NAMES[name]}</MenuItem>);
      default:
        return <MenuItem value='error'>Unknown Error</MenuItem>;
    }
  }

  function renderPlatforms() {
    switch (platforms.status) {
      case 'error':
        return <MenuItem value='error'>Error</MenuItem>;
      case 'loading':
        return <MenuItem value='loading'>Loading</MenuItem>;
      case 'success':
        return mustExist(platforms.data).map((name) => <MenuItem key={name} value={name}>{PLATFORM_NAMES[name]}</MenuItem>);
      default:
        return <MenuItem value='error'>Unknown Error</MenuItem>;
    }
  }

  return <Box>
    <Stack spacing={2}>
      <Stack direction='row' spacing={2}>
        <Select
          value={scheduler}
          label="Scheduler"
          onChange={(event) => {
            setScheduler(event.target.value);
          }}
        >
          {renderSchedulers()}
        </Select>
        <Select
          value={platform}
          label="Platform"
          onChange={(event) => {
            setPlatform(event.target.value);
          }}
        >
          {renderPlatforms()}
        </Select>
      </Stack>
      <ImageControl params={params} onChange={(newParams) => {
        setParams(newParams);
      }} />
      <TextField label="Prompt" variant="outlined" value={prompt} onChange={(event) => {
        setPrompt(event.target.value);
      }} />
      <Button onClick={() => generate.mutate()}>Generate</Button>
      {renderImage()}
    </Stack>
  </Box>;
}
