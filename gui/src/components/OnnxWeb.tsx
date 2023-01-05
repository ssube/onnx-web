import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Box, Container, Tab, Typography } from '@mui/material';
import * as React from 'react';

export function OnnxWeb() {
  const [tab, setTab] = React.useState('1');

  return (
    <div>
      <Container>
        <Box sx={{ my: 4 }}>
          <Typography variant='h3' gutterBottom>
            ONNX Web GUI
          </Typography>
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
          <TabPanel value="1">txt2img</TabPanel>
          <TabPanel value="2">img2img</TabPanel>
          <TabPanel value="3">settings</TabPanel>
        </TabContext>
      </Container>
    </div>
  );
}
