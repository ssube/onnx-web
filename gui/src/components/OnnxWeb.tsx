import { doesExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, Divider, Tab } from '@mui/material';
import * as React from 'react';
import { useHash } from 'react-use/lib/useHash';

import { ModelControl } from './control/ModelControl.js';
import { ImageHistory } from './ImageHistory.js';
import { Logo } from './Logo.js';
import { Blend } from './tab/Blend.js';
import { Img2Img } from './tab/Img2Img.js';
import { Inpaint } from './tab/Inpaint.js';
import { Settings } from './tab/Settings.js';
import { Txt2Img } from './tab/Txt2Img.js';
import { Upscale } from './tab/Upscale.js';

const REMOVE_HASH = /^#?(.*)$/;
const TAB_LABELS = [
  'txt2img',
  'img2img',
  'inpaint',
  'upscale',
  'blend',
  'settings',
];

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

    return TAB_LABELS[0];
  }

  return (
    <Container>
      <Box sx={{ my: 4 }}>
        <Logo />
      </Box>
      <Box sx={{ mx: 4, my: 4 }}>
        <ModelControl />
      </Box>
      <TabContext value={tab()}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList onChange={(_e, idx) => {
            setHash(idx);
          }}>
            {TAB_LABELS.map((name) => <Tab key={name} label={name} value={name} />)}
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
        <TabPanel value='blend'>
          <Blend />
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
