import { doesExist, mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, Divider, Link, PaletteMode, Tab, Typography, useMediaQuery } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import * as React from 'react';
import { useHash } from 'react-use/lib/useHash';

import { useStore } from 'zustand';
import { ModelControl } from './control/ModelControl.js';
import { ImageHistory } from './ImageHistory.js';
import { Blend } from './tab/Blend.js';
import { Img2Img } from './tab/Img2Img.js';
import { Inpaint } from './tab/Inpaint.js';
import { Settings } from './tab/Settings.js';
import { Txt2Img } from './tab/Txt2Img.js';
import { Upscale } from './tab/Upscale.js';
import { StateContext } from '../state.js';

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
  /* checks for system light/dark mode preference */
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const state = useStore(mustExist(React.useContext(StateContext)));

  const theme = React.useMemo(
    () => {
      if (state.theme === '') {
        if (prefersDarkMode) {
          return createTheme({
            palette: {
              mode: 'dark'
            },
          });
        } else {
          return createTheme({
            palette: {
              mode: 'light'
            },
          });
        }
      } else {
        return createTheme({
          palette: {
            mode: state.theme as PaletteMode
          },
        });
      }
    },
    [prefersDarkMode, state.theme],
  );

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
    <ThemeProvider theme={theme}>
      <CssBaseline />
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
    </ThemeProvider>
  );
}
