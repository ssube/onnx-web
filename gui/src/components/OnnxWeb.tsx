import { mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, CssBaseline, Divider, Tab, useMediaQuery } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import * as React from 'react';
import { useContext, useMemo } from 'react';
import { useHash } from 'react-use/lib/useHash';
import { useStore } from 'zustand';

import { OnnxState, StateContext } from '../state/full.js';
import { ImageHistory } from './ImageHistory.js';
import { Logo } from './Logo.js';
import { Blend } from './tab/Blend.js';
import { Img2Img } from './tab/Img2Img.js';
import { Inpaint } from './tab/Inpaint.js';
import { Models } from './tab/Models.js';
import { Settings } from './tab/Settings.js';
import { Txt2Img } from './tab/Txt2Img.js';
import { Upscale } from './tab/Upscale.js';
import { getTab, getTheme, TAB_LABELS } from './utils.js';

export function OnnxWeb() {
  /* checks for system light/dark mode preference */
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const store = mustExist(useContext(StateContext));
  const stateTheme = useStore(store, selectTheme);

  const theme = useMemo(
    () => createTheme({
      palette: {
        mode: getTheme(stateTheme, prefersDarkMode),
      },
    }),
    [prefersDarkMode, stateTheme],
  );

  const [hash, setHash] = useHash();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container>
        <Box sx={{ my: 4 }}>
          <Logo />
        </Box>
        <TabContext value={getTab(hash)}>
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
          <TabPanel value='models'>
            <Models />
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

export function selectTheme(state: OnnxState) {
  return state.theme;
}
