import { doesExist, mustExist } from '@apextoaster/js-utils';
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

export interface Img2ImgProps {
  client: ApiClient;
  config: Config;

  model: string;
  platform: string;
}

export function Img2Img(props: Img2ImgProps) {
  const { client, config, model, platform } = props;

  async function uploadSource() {
    return client.img2img({
      ...params,
      model,
      platform,
      scheduler,
      strength,
      source: mustExist(source), // TODO: show an error if this doesn't exist
    });
  }

  function changeSource(event: React.ChangeEvent<HTMLInputElement>) {
    if (doesExist(event.target.files)) {
      const file = event.target.files[0];
      if (doesExist(file)) {
        setSource(file);
      }
    }
  }

  const upload = useMutation(uploadSource);
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  const [source, setSource] = useState<File>();
  const [strength, setStrength] = useState(CONFIG_DEFAULTS.strength.default);
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
      <input type='file' onChange={changeSource} />
      <ImageControl params={params} onChange={(newParams) => {
        setParams(newParams);
      }} />
      <NumericField
        decimal
        label='Strength'
        min={CONFIG_DEFAULTS.strength.min}
        max={CONFIG_DEFAULTS.strength.max}
        step={CONFIG_DEFAULTS.strength.step}
        value={strength}
        onChange={(value) => {
          setStrength(value);
        }}
      />
      <Button onClick={() => upload.mutate()}>Generate</Button>
      <MutationHistory result={upload} limit={4} element={ImageCard}
        isEqual={(a, b) => a.output === b.output}
      />
    </Stack>
  </Box>;
}
