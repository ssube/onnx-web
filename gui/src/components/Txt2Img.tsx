import { Box, Button, Stack, TextField } from '@mui/material';
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
  })

  async function getImage() {
    const image = await client.txt2img({ ...params, prompt });
    console.log(prompt, image);
    setImage(image);
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
      <Box>
        txt2img mode
      </Box>
      <ImageControl params={params} onChange={(params) => {
        setParams(params);
      }} />
      <TextField label="Prompt" variant="outlined" value={prompt} onChange={(event) => {
        setPrompt(event.target.value);
      }} />
      <Button onClick={getImage}>Generate</Button>
      {renderImage()}
    </Stack>
  </Box>;
}
