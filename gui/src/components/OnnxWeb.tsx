/* eslint-disable @typescript-eslint/no-magic-numbers */
import { mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Button, Container, CssBaseline, Divider, Stack, Tab, useMediaQuery } from '@mui/material';
import { Breakpoint, SxProps, Theme, ThemeProvider, createTheme } from '@mui/material/styles';
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
import { TAB_LABELS, getTab, getTheme } from './utils.js';
import { Motd } from '../Motd.js';

export interface OnnxWebProps {
  motd: boolean;
}

export function OnnxWeb(props: OnnxWebProps) {
  /* checks for system light/dark mode preference */
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const store = mustExist(useContext(StateContext));
  const stateTheme = useStore(store, selectTheme);
  const layout = useStore(store, selectLayout);

  const theme = useMemo(
    () => createTheme({
      palette: {
        mode: getTheme(stateTheme, prefersDarkMode),
      },
    }),
    [prefersDarkMode, stateTheme],
  );

  const [hash, setHash] = useHash();

  const historyStyle: SxProps<Theme> = {
    mx: 4,
    my: 4,
    ...LAYOUT_STYLES[layout.direction].history.style,
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth={LAYOUT_STYLES[layout.direction].container}>
        <Box sx={{ my: 4 }}>
          <Logo />
        </Box>
        {props.motd && <Motd />}
        <Stack direction={LAYOUT_STYLES[layout.direction].direction} spacing={2}>
          <Stack direction='column'>
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
          </Stack>
          <Divider flexItem variant='middle' orientation={LAYOUT_STYLES[layout.direction].divider} />
          <Box sx={historyStyle}>
            <ImageHistory width={layout.width} />
          </Box>
        </Stack>
      </Container>
    </ThemeProvider>
  );
}

export function selectTheme(state: OnnxState) {
  return state.theme;
}

export function selectLayout(state: OnnxState) {
  return {
    direction: state.layout,
    width: state.historyWidth,
  };
}

export const LAYOUT_STYLES = {
  horizontal: {
    container: false,
    direction: 'row',
    divider: 'vertical',
    history: {
      style: {
        maxHeight: '85vb',
        overflowY: 'auto',
      },
      width: 4,
    },
  },
  vertical: {
    container: 'lg' as Breakpoint,
    direction: 'column',
    divider: 'horizontal',
    history: {
      style: {},
      width: 2,
    },
  },
} as const;
