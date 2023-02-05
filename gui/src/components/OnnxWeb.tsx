import { doesExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, Divider, Link, Tab, Typography } from '@mui/material';
import * as React from 'react';
import { useHash } from 'react-use/lib/useHash';

import { ModelControl } from './control/ModelControl.js';
import { ImageHistory } from './ImageHistory.js';
import { Img2Img } from './tab/Img2Img.js';
import { Inpaint } from './tab/Inpaint.js';
import { Settings } from './tab/Settings.js';
import { Txt2Img } from './tab/Txt2Img.js';
import { Upscale } from './tab/Upscale.js';

const REMOVE_HASH = /^#?(.*)$/;

export function OnnxWeb() {
  const [hash, setHash] = useHash();

  function tab(): string {
    const match = hash.match(REMOVE_HASH);
    if (doesExist(match)) {
      const [_full, route] = Array.from(match);
      if (route.length > 0) {
        return route;
      }
    }

    return 'txt2img';
  }

  return (
    <Container>
      <Box sx={{ my: 4 }}>
        <Typography variant='h3' gutterBottom>
          <Link href='https://github.com/ssube/onnx-web' target='_blank' underline='hover'>ONNX Web</Link>
        </Typography>
      </Box>
      <Box sx={{ mx: 4, my: 4 }}>
        <ModelControl />
      </Box>
      <TabContext value={tab()}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList onChange={(_e, idx) => {
            setHash(idx);
          }}>
            <Tab label='txt2img' value='txt2img' />
            <Tab label='img2img' value='img2img' />
            <Tab label='inpaint' value='inpaint' />
            <Tab label='upscale' value='upscale' />
            <Tab label='settings' value='settings' />
          </TabList>
        </Box>
        <TabPanel value='txt2img'>
          <Txt2Img />
        </TabPanel>
        <TabPanel value='img2img'>
          <Img2Img />
        </TabPanel>
        <TabPanel value='inpaint'>
          <Inpaint />
        </TabPanel>
        <TabPanel value='upscale'>
          <Upscale />
        </TabPanel>
        <TabPanel value='settings'>
          <Settings />
        </TabPanel>
      </TabContext>
      <Divider variant='middle' />
      <Box sx={{ mx: 4, my: 4 }}>
        <ImageHistory />
      </Box>
    </Container>
  );
}
