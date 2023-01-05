import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, MenuItem, Select, Tab, Typography } from '@mui/material';
import * as React from 'react';

import { ApiClient } from '../api/client.js';
import { Txt2Img } from './Txt2Img.js';

const { useState } = React;

export interface OnnxWebProps {
  client: ApiClient;
}

export function OnnxWeb(props: OnnxWebProps) {
  const [tab, setTab] = useState('1');
  const [model, setModel] = useState('v1.5');

  return (
    <div>
      <Container>
        <Box sx={{ my: 4 }}>
          <Typography variant='h3' gutterBottom>
            ONNX Web GUI
          </Typography>
        </Box>
        <Box sx={{ my: 4 }}>
          <Select value={model} onChange={(e) => {
            setModel(e.target.value);
          }}>
            <MenuItem value='v1.4'>Stable Diffusion v1.4</MenuItem>
            <MenuItem value='v1.5'>Stable Diffusion v1.5</MenuItem>
          </Select>
        </Box>
        <TabContext value={tab}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <TabList onChange={(_e, idx) => {
              setTab(idx);
            }}>
              <Tab label="txt2img" value="1" />
              <Tab label="img2img" value="2" />
              <Tab label="settings" value="3" />
            </TabList>
          </Box>
          <TabPanel value="1">
            <Txt2Img {...props} />
          </TabPanel>
          <TabPanel value="2">
            <Box>
              img2img using {model}
            </Box>
          </TabPanel>
          <TabPanel value="3">settings for onnx-web</TabPanel>
        </TabContext>
      </Container>
    </div>
  );
}
