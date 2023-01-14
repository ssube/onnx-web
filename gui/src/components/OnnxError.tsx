import { Alert, AlertTitle, Box, Container, Stack, Typography } from '@mui/material';
import * as React from 'react';

export interface OnnxErrorProps {
  root: string;
}

export function OnnxError(props: OnnxErrorProps) {
  return (
    <Container>
      <Box sx={{ my: 4 }}>
        <Typography variant='h3' gutterBottom>
          <a href='https://github.com/ssube/onnx-web'>ONNX Web</a>
        </Typography>
      </Box>
      <Box sx={{ my: 4 }}>
        <Stack spacing={2}>
          <Alert severity='error'>
            <AlertTitle>
              Server Error
            </AlertTitle>
            Could not fetch parameters from the ONNX web API server at <code>{props.root}</code>.
          </Alert>
          <Typography variant='body1'>
            This is a web UI for running ONNX models with GPU acceleration or in software, running locally or on a
            remote machine.
          </Typography>
          <Typography variant='body1'>
            The API runs on both Linux and Windows and provides access to the major functionality of diffusers, along
            with metadata about the available models and accelerators, and the output of previous runs. Hardware
            acceleration is supported on both AMD and Nvidia, with a CPU fallback capable of running on laptop-class
            machines.
          </Typography>
          <Typography variant='body1'>
            The GUI runs in all major browsers, including on mobile devices, and allows you to select the model and
            accelerator being used, along with the prompt and other image parameters. The last few output images are
            shown below the image controls, making it easy to refer back to previous parameters or save an image from
            earlier.
          </Typography>
          <Typography variant='body1'>
            Please <a href='https://github.com/ssube/onnx-web'>visit the Github project</a> for more information and
            make sure that <a href='https://github.com/ssube/onnx-web#configuring-and-running-the-server'>your API
            server is running</a> at <a href={props.root}>{props.root}</a>.
          </Typography>
          <Typography variant='body1' gutterBottom>
            If you are trying to use a remote API server or an alternative port, you can put the address into the
            query string, like <code>{window.location.origin}?api=http://localhost:5001</code>.
          </Typography>
        </Stack>
      </Box>
    </Container>
  );
}
