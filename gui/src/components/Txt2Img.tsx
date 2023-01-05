import { Box, Button, MenuItem, Select, Stack, TextField } from '@mui/material';
import * as React from 'react';

import { ApiClient } from '../api/client.js';
import { ImageControl, ImageParams } from './ImageControl.js';

const { useState } = React;

export interface Txt2ImgProps {
  client: ApiClient;
}

export function Txt2Img(props: Txt2ImgProps) {
  const { client } = props;
  const [image, setImage] = useState('');

  const [prompt, setPrompt] = useState('an astronaut eating a hamburger');
  const [params, setParams] = useState<ImageParams>({
    cfg: 6,
    steps: 25,
    width: 512,
    height: 512,
  });
  const [scheduler, setScheduler] = useState('euler-a');

  async function getImage() {
    const data = await client.txt2img({ ...params, prompt, scheduler });
    setImage(data);
  }

  function renderImage() {
    if (image === '') {
      return <div>No image</div>;
    } else {
      return <img src={image} />;
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
        <MenuItem value='pndm'>PNDM</MenuItem>
        <MenuItem value='lms-discrete'>LMS</MenuItem>
        <MenuItem value='euler'>Euler</MenuItem>
        <MenuItem value='euler-a'>Euler A</MenuItem>
        <MenuItem value='dpm-multi'>DPM</MenuItem>
      </Select>
      <ImageControl params={params} onChange={(newParams) => {
        setParams(newParams);
      }} />
      <TextField label="Prompt" variant="outlined" value={prompt} onChange={(event) => {
        setPrompt(event.target.value);
      }} />
      <Button onClick={getImage}>Generate</Button>
      {renderImage()}
    </Stack>
  </Box>;
}
