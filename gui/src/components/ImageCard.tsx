import { doesExist, Maybe, mustDefault, mustExist } from '@apextoaster/js-utils';
import { Blender, Brush, ContentCopy, Delete, Download, ZoomOutMap } from '@mui/icons-material';
import { Box, Card, CardContent, CardMedia, Grid, IconButton, Menu, MenuItem, Paper, Tooltip } from '@mui/material';
import * as React from 'react';
import { useContext, useState } from 'react';
import { useHash } from 'react-use/lib/useHash';
import { useStore } from 'zustand';

import { ImageResponse } from '../client.js';
import { BLEND_SOURCES, ConfigContext, StateContext } from '../state.js';
import { MODEL_LABELS, SCHEDULER_LABELS } from '../strings.js';
import { range, visibleIndex } from '../utils.js';

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

  const [_hash, setHash] = useHash();
  const [anchor, setAnchor] = useState<Maybe<HTMLElement>>();

  const config = mustExist(useContext(ConfigContext));
  const state = mustExist(useContext(StateContext));
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setImg2Img = useStore(state, (s) => s.setImg2Img);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setInpaint = useStore(state, (s) => s.setInpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setUpscale = useStore(state, (s) => s.setUpscaleTab);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setBlend = useStore(state, (s) => s.setBlend);

  async function loadSource() {
    const req = await fetch(output[0].url);
    return req.blob();
  }

  async function copySourceToImg2Img() {
    const blob = await loadSource();
    setImg2Img({
      source: blob,
    });
    setHash('img2img');
  }

  async function copySourceToInpaint() {
    const blob = await loadSource();
    setInpaint({
      source: blob,
    });
    setHash('inpaint');
  }

  async function copySourceToUpscale() {
    const blob = await loadSource();
    setUpscale({
      source: blob,
    });
    setHash('upscale');
  }

  async function copySourceToBlend(idx: number) {
    const blob = await loadSource();
    const sources = mustDefault(state.getState().blend.sources, []);
    const newSources = [...sources];
    newSources[idx] = blob;
    setBlend({
      sources: newSources,
    });
    setHash('blend');
  }

  function deleteImage() {
    if (doesExist(props.onDelete)) {
      props.onDelete(value);
    }
  }

  function downloadImage() {
    window.open(output[0].url, '_blank');
  }

  function close() {
    setAnchor(undefined);
  }

  const model = mustDefault(MODEL_LABELS[params.model], params.model);
  const scheduler = mustDefault(SCHEDULER_LABELS[params.scheduler], params.scheduler);

  return <Card sx={{ maxWidth: config.params.width.default }} elevation={2}>
    <CardMedia sx={{ height: config.params.height.default }}
      component='img'
      image={output[0].url}
      title={params.prompt}
    />
    <CardContent>
      <Box textAlign='center'>
        <Grid container spacing={2}>
          <GridItem xs={4}>Model: {model}</GridItem>
          <GridItem xs={4}>Scheduler: {scheduler}</GridItem>
          <GridItem xs={4}>Seed: {params.seed}</GridItem>
          <GridItem xs={4}>CFG: {params.cfg}</GridItem>
          <GridItem xs={4}>Steps: {params.steps}</GridItem>
          <GridItem xs={4}>Size: {size.width}x{size.height}</GridItem>
          <GridItem xs={12}>
            <Box textAlign='left'>{params.prompt}</Box>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title='Save'>
              <IconButton onClick={downloadImage}>
                <Download />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title='Img2img'>
              <IconButton onClick={copySourceToImg2Img}>
                <ContentCopy />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title='Inpaint'>
              <IconButton onClick={copySourceToInpaint}>
                <Brush />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title='Upscale'>
              <IconButton onClick={copySourceToUpscale}>
                <ZoomOutMap />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title='Blend'>
              <IconButton onClick={(event) => {
                setAnchor(event.currentTarget);
              }}>
                <Blender />
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={anchor}
              open={doesExist(anchor)}
              onClose={close}
            >
              {range(BLEND_SOURCES).map((idx) => <MenuItem key={idx} onClick={() => {
                copySourceToBlend(idx).catch((err) => {
                  // TODO
                });
                close();
              }}>{visibleIndex(idx)}</MenuItem>)}
            </Menu>
          </GridItem>
          <GridItem xs={2}>
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
