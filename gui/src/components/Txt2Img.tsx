import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQuery } from 'react-query';

import { ApiClient, BaseImgParams } from '../api/client.js';
import { Config, CONFIG_DEFAULTS, STALE_TIME } from '../config.js';
import { SCHEDULER_LABELS } from '../strings.js';
import { ImageCard } from './ImageCard.js';
import { ImageControl } from './ImageControl.js';
import { MutationHistory } from './MutationHistory.js';
import { NumericField } from './NumericField.js';
import { QueryList } from './QueryList.js';

const { useState } = React;

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
      scheduler,
      height,
      width,
    });
  }

  const generate = useMutation(generateImage);
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  const [height, setHeight] = useState(CONFIG_DEFAULTS.height.default);
  const [width, setWidth] = useState(CONFIG_DEFAULTS.width.default);
  const [params, setParams] = useState<BaseImgParams>({
    cfg: CONFIG_DEFAULTS.cfg.default,
    seed: CONFIG_DEFAULTS.seed.default,
    steps: CONFIG_DEFAULTS.steps.default,
    prompt: config.default.prompt,
  });
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
      <Stack direction='row' spacing={4}>
        <NumericField
          label='Width'
          min={CONFIG_DEFAULTS.width.min}
          max={CONFIG_DEFAULTS.width.max}
          step={CONFIG_DEFAULTS.width.step}
          value={width}
          onChange={(value) => {
            setWidth(value);
          }}
        />
        <NumericField
          label='Height'
          min={CONFIG_DEFAULTS.height.min}
          max={CONFIG_DEFAULTS.height.max}
          step={CONFIG_DEFAULTS.height.step}
          value={height}
          onChange={(value) => {
            setHeight(value);
          }}
        />
      </Stack>
      <Button onClick={() => generate.mutate()}>Generate</Button>
      <MutationHistory result={generate} limit={4} element={ImageCard}
        isEqual={(a, b) => a.output === b.output}
      />
    </Stack>
  </Box>;
}
