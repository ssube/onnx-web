import { mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, CssBaseline, Divider, Stack, Tab, useMediaQuery } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { Allotment } from 'allotment';
import * as React from 'react';
import { useContext, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useHash } from 'react-use/lib/useHash';
import { useStore } from 'zustand';

import { LAYOUT_MIN, LAYOUT_PROPORTIONS, LAYOUT_STYLES, STANDARD_MARGIN, STANDARD_SPACING } from '../constants.js';
import { Motd } from '../Motd.js';
import { OnnxState, StateContext } from '../state/full.js';
import { Layout } from '../state/settings.js';
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

import 'allotment/dist/style.css';
import './main.css';

export interface OnnxWebProps {
  motd: boolean;
}

export function OnnxWeb(props: OnnxWebProps) {
  /* checks for system light/dark mode preference */
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const store = mustExist(useContext(StateContext));
  const stateTheme = useStore(store, selectTheme);
  const historyWidth = useStore(store, selectHistoryWidth);
  const direction = useStore(store, selectDirection);

  const layout = LAYOUT_STYLES[direction];

  const theme = useMemo(
    () => createTheme({
      palette: {
        mode: getTheme(stateTheme, prefersDarkMode),
      },
    }),
    [prefersDarkMode, stateTheme],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth={layout.container}>
        <Box sx={{ my: STANDARD_MARGIN }}>
          <Logo />
        </Box>
        {props.motd && <Motd />}
        {renderBody(direction, historyWidth)}
      </Container>
    </ThemeProvider>
  );
}

export function selectTheme(state: OnnxState) {
  return state.theme;
}

export function selectDirection(state: OnnxState) {
  return state.layout;
}

export function selectHistoryWidth(state: OnnxState) {
  return state.historyWidth;
}

function renderBody(direction: Layout, historyWidth: number) {
  if (direction === 'vertical') {
    return <VerticalBody direction={direction} width={historyWidth} />;
  } else {
    return <HorizontalBody direction={direction} width={historyWidth} />;
  }
}

// used for both horizontal and vertical
export interface BodyProps {
  direction: Layout;
  width: number;
}

export function HorizontalBody(props: BodyProps) {
  const layout = LAYOUT_STYLES[props.direction];

  return <Allotment
    className='body-allotment'
    defaultSizes={LAYOUT_PROPORTIONS}
    minSize={LAYOUT_MIN}
    separator
    snap
  >
    <TabGroup direction={props.direction} />
    <Box className='box-history' sx={layout.history.style}>
      <ImageHistory width={props.width} />
    </Box>
  </Allotment>;
}

export function VerticalBody(props: BodyProps) {
  const layout = LAYOUT_STYLES[props.direction];

  return <Stack direction={layout.direction} spacing={STANDARD_SPACING}>
    <TabGroup direction={props.direction} />
    <Divider flexItem variant='middle' orientation={layout.divider} />
    <Box className='box-history' sx={layout.history.style}>
      <ImageHistory width={props.width} />
    </Box>
  </Stack>;
}

export interface TabGroupProps {
  direction: Layout;
}

export function TabGroup(props: TabGroupProps) {
  const layout = LAYOUT_STYLES[props.direction];

  const [hash, setHash] = useHash();
  const { t } = useTranslation();

  return <Stack direction='column' minWidth={layout.control.width} sx={{ mx: STANDARD_MARGIN }}>
    <TabContext value={getTab(hash)}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <TabList onChange={(_e, idx) => {
          setHash(idx);
        }}>
          {TAB_LABELS.map((name) => <Tab key={name} label={t(`tab.${name}`)} value={name} />)}
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
  </Stack>;
}
