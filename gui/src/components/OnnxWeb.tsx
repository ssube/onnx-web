import { mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList } from '@mui/lab';
import { Box, Container, Divider, PaletteMode, Tab, useMediaQuery, CssBaseline } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import * as React from 'react';
import { useContext, useMemo } from 'react';
import { useHash } from 'react-use/lib/useHash';

import { useStore } from 'zustand';
import { ModelControl } from './control/ModelControl.js';
import { ImageHistory } from './ImageHistory.js';
import { Logo } from './Logo.js';
import { StateContext } from '../state.js';
import { getTheme, getTab, TAB_LABELS } from './utils.js';

export function OnnxWeb() {
  /* checks for system light/dark mode preference */
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const state = useStore(mustExist(useContext(StateContext)));

  const theme = useMemo(
    () => createTheme({
      palette: {
        mode: getTheme(state.theme, prefersDarkMode) as PaletteMode
      }
    }),
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