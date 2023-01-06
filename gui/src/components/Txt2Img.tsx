import { Box, Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useMutation, useQuery } from 'react-query';

import { ApiClient } from '../api/client.js';
import { ImageCard } from './ImageCard.js';
import { ImageControl, ImageParams } from './ImageControl.js';
import { MutationHistory } from './MutationHistory.js';
import { QueryList } from './QueryList.js';

const { useState } = React;

export const STALE_TIME = 3_000;

// TODO: set up i18next
const PLATFORM_LABELS: Record<string, string> = {
  amd: 'AMD GPU',
  cpu: 'CPU',
};

const SCHEDULER_LABELS: Record<string, string> = {
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
  const { client, model } = props;

  async function generateImage() {
    return client.txt2img({
      ...params,
      model,
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
  const [platform, setPlatform] = useState('amd');

  return <Box>
    <Stack spacing={2}>
      <Stack direction='row' spacing={2}>
        <QueryList result={schedulers} value={scheduler} labels={SCHEDULER_LABELS}
          onChange={(value) => {
            setScheduler(value);
          }}
        />
        <QueryList result={platforms} value={platform} labels={PLATFORM_LABELS}
          onChange={(value) => {
            setPlatform(value);
          }}
        />
      </Stack>
      <ImageControl params={params} onChange={(newParams) => {
        setParams(newParams);
      }} />
      <TextField label="Prompt" variant="outlined" value={prompt} onChange={(event) => {
        setPrompt(event.target.value);
      }} />
      <Button onClick={() => generate.mutate()}>Generate</Button>
      <MutationHistory result={generate} limit={4} element={ImageCard}
        isPresent={(list, item) => list.some((other) => item.output === other.output)}
      />
    </Stack>
  </Box>;
}
