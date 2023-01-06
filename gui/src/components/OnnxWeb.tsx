import { mustExist } from '@apextoaster/js-utils';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, MenuItem, Select, Tab, Typography } from '@mui/material';
import * as React from 'react';
import { useQuery } from 'react-query';

import { ApiClient } from '../api/client.js';
import { STALE_TIME, Txt2Img } from './Txt2Img.js';

const { useState } = React;

export interface OnnxWebProps {
  client: ApiClient;
}

export function OnnxWeb(props: OnnxWebProps) {
  const [tab, setTab] = useState('1');
  const [model, setModel] = useState('stable-diffusion-onnx-v1-5');

  const models = useQuery('models', async () => props.client.models(), {
    staleTime: STALE_TIME,
  });

  function renderModels() {
    switch (models.status) {
      case 'error':
        return <MenuItem value='error'>Error</MenuItem>;
      case 'loading':
        return <MenuItem value='loading'>Loading</MenuItem>;
      case 'success':
        return mustExist(models.data).map((name) => <MenuItem key={name} value={name}>{name}</MenuItem>);
      default:
        return <MenuItem value='error'>Unknown Error</MenuItem>;
    }
  }

  return (
    <div>
      <Container>
        <Box sx={{ my: 4 }}>
          <Typography variant='h3' gutterBottom>
            ONNX Web
          </Typography>
        </Box>
        <Box sx={{ my: 4 }}>
          <Select value={model} onChange={(e) => {
            setModel(e.target.value);
          }}>
            {renderModels()}
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
            <Txt2Img client={props.client} model={model} />
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
