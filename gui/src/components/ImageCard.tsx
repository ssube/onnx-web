import { doesExist, mustExist } from '@apextoaster/js-utils';
import { Brush, ContentCopy, ContentCopyTwoTone, Delete, Download } from '@mui/icons-material';
import { Box, Button, Card, CardContent, CardMedia, Grid, IconButton, Paper, Tooltip } from '@mui/material';
import * as React from 'react';
import { useContext } from 'react';
import { useStore } from 'zustand';

import { ImageResponse } from '../client.js';
import { ConfigContext, StateContext } from '../state.js';

export interface ImageCardProps {
  value: ImageResponse;

  onDelete?: (key: ImageResponse) => void;
}

export function GridItem(props: { xs: number; children: React.ReactNode }) {
  return <Grid item xs={props.xs}>
    <Paper elevation={0} sx={{ padding: 1 }}>{props.children}</Paper>
  </Grid>;
}

export function ImageCard(props: ImageCardProps) {
  const { value } = props;
  const { params, output, size } = value;

  const config = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setImg2Img = useStore(state, (s) => s.setImg2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setInpaint = useStore(state, (s) => s.setInpaint);

  async function loadSource() {
    const req = await fetch(output.url);
    return req.blob();
  }

  async function copySourceToImg2Img() {
    const blob = await loadSource();
    setImg2Img({
      source: blob,
    });
  }

  async function copySourceToInpaint() {
    const blob = await loadSource();
    setInpaint({
      source: blob,
    });
  }

  function deleteImage() {
    if (doesExist(props.onDelete)) {
      props.onDelete(value);
    }
  }

  function downloadImage() {
    window.open(output.url, '_blank');
  }

  return <Card sx={{ maxWidth: config.params.width.default }} elevation={2}>
    <CardMedia sx={{ height: config.params.height.default }}
      component='img'
      image={output.url}
      title={params.prompt}
    />
    <CardContent>
      <Box textAlign='center'>
        <Grid container spacing={2}>
          <GridItem xs={4}>CFG: {params.cfg}</GridItem>
          <GridItem xs={4}>Steps: {params.steps}</GridItem>
          <GridItem xs={4}>Size: {size.width}x{size.height}</GridItem>
          <GridItem xs={4}>Seed: {params.seed}</GridItem>
          <GridItem xs={8}>Scheduler: {params.scheduler}</GridItem>
          <GridItem xs={12}>
            <Box textAlign='left'>{params.prompt}</Box>
          </GridItem>
          <GridItem xs={3}>
            <Tooltip title='Save'>
              <IconButton onClick={downloadImage}>
                <Download />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={3}>
            <Tooltip title='Img2img'>
              <IconButton onClick={copySourceToImg2Img}>
                <ContentCopy />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={3}>
            <Tooltip title='Inpaint'>
              <IconButton onClick={copySourceToInpaint}>
                <Brush />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={3}>
            <Tooltip title='Delete'>
              <IconButton onClick={deleteImage}>
                <Delete />
              </IconButton>
            </Tooltip>
          </GridItem>
        </Grid>
      </Box>
    </CardContent>
  </Card>;
}
