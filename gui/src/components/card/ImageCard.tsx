import { doesExist, Maybe, mustDefault, mustExist } from '@apextoaster/js-utils';
import { ArrowLeft, ArrowRight, Blender, Brush, ContentCopy, Delete, Download, ZoomOutMap } from '@mui/icons-material';
import { Box, Card, CardContent, CardMedia, Grid, IconButton, Menu, MenuItem, Paper, Tooltip } from '@mui/material';
import * as React from 'react';
import { useContext, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useHash } from 'react-use/lib/useHash';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

import { BLEND_SOURCES, ConfigContext, OnnxState, StateContext } from '../../state.js';
import { ImageResponse } from '../../types/api.js';
import { range, visibleIndex } from '../../utils.js';

export interface ImageCardProps {
  image: ImageResponse;

  onDelete?: (key: ImageResponse) => void;
}

export function GridItem(props: { xs: number; children: React.ReactNode }) {
  return <Grid item xs={props.xs}>
    <Paper elevation={0} sx={{ padding: 1 }}>{props.children}</Paper>
  </Grid>;
}

export function ImageCard(props: ImageCardProps) {
  const { image } = props;
  const { params, outputs, size } = image;

  const [_hash, setHash] = useHash();
  const [blendAnchor, setBlendAnchor] = useState<Maybe<HTMLElement>>();
  const [saveAnchor, setSaveAnchor] = useState<Maybe<HTMLElement>>();

  const config = mustExist(useContext(ConfigContext));
  const store = mustExist(useContext(StateContext));
  const { setBlend, setImg2Img, setInpaint, setUpscale } = useStore(store, selectActions, shallow);

  async function loadSource() {
    const req = await fetch(outputs[index].url);
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
    const sources = mustDefault(store.getState().blend.sources, []);
    const newSources = [...sources];
    newSources[idx] = blob;
    setBlend({
      sources: newSources,
    });
    setHash('blend');
  }

  function deleteImage() {
    if (doesExist(props.onDelete)) {
      props.onDelete(image);
    }
  }

  function downloadImage() {
    window.open(outputs[index].url, '_blank');
    close();
  }

  function downloadMetadata() {
    window.open(outputs[index].url + '.json', '_blank');
    close();
  }

  function close() {
    // TODO: split these up
    setBlendAnchor(undefined);
    setSaveAnchor(undefined);
  }

  const [index, setIndex] = useState(0);
  const { t } = useTranslation();

  function getLabel(key: string, name: string) {
    return mustDefault(t(`${key}.${name}`), name);
  }

  const model = getLabel('model', params.model);
  const scheduler = getLabel('scheduler', params.scheduler);

  return <Card sx={{ maxWidth: config.params.width.default }} elevation={2}>
    <CardMedia sx={{ height: config.params.height.default }}
      component='img'
      image={outputs[index].url}
      title={params.prompt}
    />
    <CardContent>
      <Box textAlign='center'>
        <Grid container spacing={2}>
          <GridItem xs={4}>
            <Tooltip title={t('tooltip.previous')}>
              <IconButton onClick={() => {
                const prevIndex = index - 1;
                if (prevIndex < 0) {
                  setIndex(outputs.length + prevIndex);
                } else {
                  setIndex(prevIndex);
                }
              }}>
                <ArrowLeft />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={4}>
            {visibleIndex(index)} of {outputs.length}
          </GridItem>
          <GridItem xs={4}>
            <Tooltip title={t('tooltip.next')}>
              <IconButton onClick={() => {
                setIndex((index + 1) % outputs.length);
              }}>
                <ArrowRight />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={4}>{t('modelType.diffusion', {count: 1})}: {model}</GridItem>
          <GridItem xs={4}>{t('parameter.scheduler')}: {scheduler}</GridItem>
          <GridItem xs={4}>{t('parameter.seed')}: {params.seed}</GridItem>
          <GridItem xs={4}>{t('parameter.cfg')}: {params.cfg}</GridItem>
          <GridItem xs={4}>{t('parameter.steps')}: {params.steps}</GridItem>
          <GridItem xs={4}>{t('parameter.size')}: {size.width}x{size.height}</GridItem>
          <GridItem xs={12}>
            <Box textAlign='left'>{params.prompt}</Box>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title={t('tooltip.save')}>
              <IconButton onClick={(event) => {
                setSaveAnchor(event.currentTarget);
              }}>
                <Download />
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={saveAnchor}
              open={doesExist(saveAnchor)}
              onClose={close}
            >
              <MenuItem key='save-image' onClick={downloadImage}>{t('save.image')}</MenuItem>
              <MenuItem key='save-metadata' onClick={downloadMetadata}>{t('save.metadata')}</MenuItem>
            </Menu>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title={t('tab.img2img')}>
              <IconButton onClick={copySourceToImg2Img}>
                <ContentCopy />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title={t('tab.inpaint')}>
              <IconButton onClick={copySourceToInpaint}>
                <Brush />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title={t('tab.upscale')}>
              <IconButton onClick={copySourceToUpscale}>
                <ZoomOutMap />
              </IconButton>
            </Tooltip>
          </GridItem>
          <GridItem xs={2}>
            <Tooltip title={t('tab.blend')}>
              <IconButton onClick={(event) => {
                setBlendAnchor(event.currentTarget);
              }}>
                <Blender />
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={blendAnchor}
              open={doesExist(blendAnchor)}
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
            <Tooltip title={t('tooltip.delete')}>
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

export function selectActions(state: OnnxState) {
  return {
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setBlend: state.setBlend,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setImg2Img: state.setImg2Img,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setInpaint: state.setInpaint,
    // eslint-disable-next-line @typescript-eslint/unbound-method
    setUpscale: state.setUpscale,
  };
}
