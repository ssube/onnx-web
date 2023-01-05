import { Box, Button, TextField } from "@mui/material";
import * as React from 'react';
import { ApiClient } from "../api/client";

const { useEffect, useState } = React;

export interface Txt2ImgProps {
  client: ApiClient;
}

export function Txt2Img(props: Txt2ImgProps) {
  const { client } = props;
  const [image, setImage] = useState('');

  const [cfg, setCfg] = useState(5);
  const [prompt, setPrompt] = useState('an astronaut eating a hamburger');
  const [steps, setSteps] = useState(20);

  async function getImage() {
    const image = await client.txt2img({ prompt, cfg, steps });
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
    <TextField label="CFG" variant="outlined" type="number" inputProps={{ min: 5, max: 30 }} value={cfg} onChange={(event) => {
      setCfg(parseInt(event.target.value, 10));
    }} />
    <TextField label="Steps" variant="outlined" type="number" inputProps={{ min: 15, max: 150 }} value={steps} onChange={(event) => {
      setSteps(parseInt(event.target.value, 10));
    }} />
    <TextField label="Prompt" variant="outlined" value={prompt} onChange={(event) => {
      console.log('changing prompt', event.target.value);
      setPrompt(event.target.value);
    }} />
    <Button onClick={getImage}>Generate</Button>
    {renderImage()}
  </Box>;
}
