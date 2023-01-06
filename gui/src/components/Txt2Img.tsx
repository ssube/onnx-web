import { Box, Button, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useMutation, useQuery } from 'react-query';

import { ApiClient } from '../api/client.js';
import { Config } from '../config.js';
import { SCHEDULER_LABELS } from '../strings.js';
import { ImageCard } from './ImageCard.js';
import { ImageControl, ImageParams } from './ImageControl.js';
import { MutationHistory } from './MutationHistory.js';
import { QueryList } from './QueryList.js';

const { useState } = React;

export const STALE_TIME = 3_000;
export interface Txt2ImgProps {
  client: ApiClient;
  config: Config;

  model: string;
  platform: string;
}

export function Txt2Img(props: Txt2ImgProps) {
  const { client, config, model, platform } = props;

  async function generateImage() {
    return client.txt2img({
      ...params,
      model,
      platform,
      prompt,
      scheduler,
    });
  }

  const generate = useMutation(generateImage);
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  const [params, setParams] = useState<ImageParams>({
    cfg: 6,
    steps: 25,
    width: 512,
    height: 512,
  });
  const [prompt, setPrompt] = useState(config.default.prompt);
  const [scheduler, setScheduler] = useState(config.default.scheduler);

  return <Box>
    <Stack spacing={2}>
      <Stack direction='row' spacing={2}>
        <QueryList
          id='schedulers'
          labels={SCHEDULER_LABELS}
          name='Scheduler'
          result={schedulers}
          value={scheduler}
          onChange={(value) => {
            setScheduler(value);
          }}
        />
      </Stack>
      <ImageControl params={params} onChange={(newParams) => {
        setParams(newParams);
      }} />
      <TextField label='Prompt' variant='outlined' value={prompt} onChange={(event) => {
        setPrompt(event.target.value);
      }} />
      <Button onClick={() => generate.mutate()}>Generate</Button>
      <MutationHistory result={generate} limit={4} element={ImageCard}
        isPresent={(list, item) => list.some((other) => item.output === other.output)}
      />
    </Stack>
  </Box>;
}
