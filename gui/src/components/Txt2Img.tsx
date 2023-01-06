import { Box, Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';
import { useMutation } from 'react-query';

import { ApiClient } from '../api/client.js';
import { ImageControl, ImageParams } from './ImageControl.js';

const { useState } = React;

export interface Txt2ImgProps {
  client: ApiClient;
}

export function Txt2Img(props: Txt2ImgProps) {
  const { client } = props;

  async function generateImage() {
    return client.txt2img({ ...params, prompt, scheduler });
  }

  const generate = useMutation(generateImage);

  const [prompt, setPrompt] = useState('an astronaut eating a hamburger');
  const [params, setParams] = useState<ImageParams>({
    cfg: 6,
    steps: 25,
    width: 512,
    height: 512,
  });
  const [scheduler, setScheduler] = useState('euler-a');

  function renderImage() {
    switch (generate.status) {
      case 'error':
        if (generate.error instanceof Error) {
          return <div>{generate.error.message}</div>;
        } else {
          return <div>Unknown error generating image.</div>;
        }
      case 'success':
        return <img src={generate.data.output} />;
      default:
        return <div>No result. Press Generate.</div>;
    }
  }

  return <Box>
    <Stack spacing={2}>
      <Select
        value={scheduler}
        label="Scheduler"
        onChange={(event) => {
          setScheduler(event.target.value);
        }}
      >
        <MenuItem value='ddim'>DDIM</MenuItem>
        <MenuItem value='ddpm'>DDPM</MenuItem>
        <MenuItem value='dpm-multi'>DPM Multistep</MenuItem>
        <MenuItem value='euler'>Euler</MenuItem>
        <MenuItem value='euler-a'>Euler Ancestral</MenuItem>
        <MenuItem value='lms-discrete'>LMS Discrete</MenuItem>
        <MenuItem value='pndm'>PNDM</MenuItem>
      </Select>
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
