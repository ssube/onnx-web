/* eslint-disable @typescript-eslint/no-magic-numbers */
import { mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, CssBaseline, Divider, Stack, Tab, useMediaQuery } from '@mui/material';
import { Breakpoint, SxProps, Theme, ThemeProvider, createTheme } from '@mui/material/styles';
import { Allotment } from 'allotment';
import * as React from 'react';
import { useContext, useMemo } from 'react';
import { useHash } from 'react-use/lib/useHash';
import { useStore } from 'zustand';
import { shallow } from 'zustand/shallow';

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
import { TAB_LABELS, getTab, getTheme } from './utils.js';

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
  const layout = useStore(store, selectLayout, shallow);

  const theme = useMemo(
    () => createTheme({
      palette: {
        mode: getTheme(stateTheme, prefersDarkMode),
      },
    }),
    [prefersDarkMode, stateTheme],
  );

  const historyStyle: SxProps<Theme> = LAYOUT_STYLES[layout.direction].history.style;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth={LAYOUT_STYLES[layout.direction].container}>
        <Box sx={{ my: 4 }}>
          <Logo />
        </Box>
        {props.motd && <Motd />}
        {renderBody(layout, historyStyle)}
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
    control: {
      width: '30%',
    },
    direction: 'row',
    divider: 'vertical',
    history: {
      style: {
        marginLeft: 4,
        maxHeight: '85vb',
        overflowY: 'auto',
      },
      width: 4,
    },
  },
  vertical: {
    container: 'lg' as Breakpoint,
    control: {
      width: undefined,
    },
    direction: 'column',
    divider: 'horizontal',
    history: {
      style: {
        mx: 4,
        my: 4,
      },
      width: 2,
    },
  },
} as const;

function renderBody(layout: ReturnType<typeof selectLayout>, historyStyle: SxProps<Theme>) {
  if (layout.direction === 'vertical') {
    return <VerticalBody {...layout} style={historyStyle} />;
  } else {
    return <HorizontalBody {...layout} style={historyStyle} />;
  }
}

// used for both horizontal and vertical
export interface BodyProps {
  direction: Layout;
  style: SxProps<Theme>;
  width: number;
}

export function HorizontalBody(props: BodyProps) {
  const layout = LAYOUT_STYLES[props.direction];

  return <Allotment separator className='body-allotment' minSize={300}>
    <TabGroup direction={props.direction} />
    <Box sx={layout.history.style}>
      <ImageHistory width={props.width} />
    </Box>
  </Allotment>;
}

export function VerticalBody(props: BodyProps) {
  const layout = LAYOUT_STYLES[props.direction];

  return <Stack direction={layout.direction} spacing={2}>
    <TabGroup direction={props.direction} />
    <Divider flexItem variant='middle' orientation={layout.divider} />
    <Box sx={layout.history.style}>
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

  return <Stack direction='column' minWidth={layout.control.width} sx={{ mx: 4 }}>
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
  </Stack>;
}
