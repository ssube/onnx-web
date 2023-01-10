import { mustExist } from '@apextoaster/js-utils';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQuery } from 'react-query';

import { ApiClient, BaseImgParams, paramsFromConfig } from '../api/client.js';
import { ConfigParams, IMAGE_FILTER, STALE_TIME } from '../config.js';
import { SCHEDULER_LABELS } from '../strings.js';
import { ImageCard } from './ImageCard.js';
import { ImageControl } from './ImageControl.js';
import { ImageInput } from './ImageInput.js';
import { MutationHistory } from './MutationHistory.js';
import { NumericField } from './NumericField.js';
import { QueryList } from './QueryList.js';

const { useState } = React;

export interface Img2ImgProps {
  client: ApiClient;
  config: ConfigParams;

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

  const upload = useMutation(uploadSource);
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  const [source, setSource] = useState<File>();
  const [strength, setStrength] = useState(config.strength.default);
  const [params, setParams] = useState<BaseImgParams>(paramsFromConfig(config));
  const [scheduler, setScheduler] = useState(config.scheduler.default);

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
      <ImageInput filter={IMAGE_FILTER} label='Source' onChange={setSource} />
      <ImageControl config={config} params={params} onChange={(newParams) => {
        setParams(newParams);
      }} />
      <NumericField
        decimal
        label='Strength'
        min={config.strength.min}
        max={config.strength.max}
        step={config.strength.step}
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
