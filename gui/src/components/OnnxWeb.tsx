import { doesExist, mustExist } from '@apextoaster/js-utils';
import { doesExist, mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, Divider, PaletteMode, Tab, useMediaQuery, CssBaseline } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { Box, Container, Divider, PaletteMode, Tab, useMediaQuery, CssBaseline } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import * as React from 'react';
import { useHash } from 'react-use/lib/useHash';

import { useStore } from 'zustand';
import { useStore } from 'zustand';
import { ModelControl } from './control/ModelControl.js';
import { ImageHistory } from './ImageHistory.js';
import { Logo } from './Logo.js';
import { Blend } from './tab/Blend.js';
import { Img2Img } from './tab/Img2Img.js';
import { Inpaint } from './tab/Inpaint.js';
import { Settings } from './tab/Settings.js';
import { Txt2Img } from './tab/Txt2Img.js';
import { Upscale } from './tab/Upscale.js';
import { StateContext } from '../state.js';
import { getTab, TAB_LABELS } from './utils.js';
import { StateContext } from '../state.js';

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

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container>
        <Box sx={{ my: 4 }}>
          <Logo />
        </Box>
        <Box sx={{ mx: 4, my: 4 }}>
          <ModelControl />
        </Box>
        <TabContext value={getTab(hash)}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <TabList onChange={(_e, idx) => {
              setHash(idx);
            }}>
              {TAB_LABELS.map((name) => <Tab key={name} label={name} value={name} />)}
            </TabList>
          </Box>
          <Box sx={{ mx: 4, my: 4 }}>
            <ModelControl />
          </Box>
        </TabContext>
        <Divider variant='middle' />
        <Box sx={{ mx: 4, my: 4 }}>
          <ImageHistory />
        </Box>
      </Container>
    </ThemeProvider>
  );
}