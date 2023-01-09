import { doesExist, mustExist } from '@apextoaster/js-utils';
import { FormatColorFill, Gradient } from '@mui/icons-material';
import { Box, Button, Stack } from '@mui/material';
import * as React from 'react';
import { useMutation, useQuery } from 'react-query';

import { ApiClient, ApiResponse, BaseImgParams, equalResponse } from '../api/client.js';
import { Config, CONFIG_DEFAULTS, DEFAULT_BRUSH, IMAGE_FILTER, STALE_TIME } from '../config.js';
import { SCHEDULER_LABELS } from '../strings.js';
import { ImageInput } from './ImageInput.js';
import { ImageCard } from './ImageCard.js';
import { ImageControl } from './ImageControl.js';
import { MutationHistory } from './MutationHistory.js';
import { NumericField } from './NumericField.js';
import { QueryList } from './QueryList.js';

const { useEffect, useRef, useState } = React;

export const FULL_CIRCLE = 2 * Math.PI;
export const PIXEL_SIZE = 4;
export const PIXEL_WEIGHT = 3;

export const COLORS = {
  black: 0,
  white: 255,
};

export const THRESHOLDS = {
  lower: 34,
  upper: 224,
};

export function floodBelow(n: number): number {
  if (n < THRESHOLDS.upper) {
    return COLORS.black;
  } else {
    return COLORS.white;
  }
}

export function floodAbove(n: number): number {
  if (n > THRESHOLDS.lower) {
    return COLORS.white;
  } else {
    return COLORS.black;
  }
}

export function grayToRGB(n: number): string {
  return `rgb(${n.toFixed(0)},${n.toFixed(0)},${n.toFixed(0)})`;
}

export interface Point {
  x: number;
  y: number;
}

export interface InpaintProps {
  client: ApiClient;
  config: Config;

  model: string;
  platform: string;
}

export function Inpaint(props: InpaintProps) {
  const { client, config, model, platform } = props;

  async function uploadSource() {
    const canvas = mustExist(canvasRef.current);
    return new Promise<ApiResponse>((res, _rej) => {
      canvas.toBlob((blob) => {
        res(client.inpaint({
          ...params,
          model,
          platform,
          scheduler,
          mask: mustExist(blob),
          source: mustExist(source),
        }));
      });
    });
  }

  function drawSource(file: File) {
    const image = new Image();
    const src = URL.createObjectURL(file);
    image.onload = () => {
      const canvas = mustExist(canvasRef.current);
      const ctx = mustExist(canvas.getContext('2d'));
      ctx.drawImage(image, 0, 0);
      URL.revokeObjectURL(src);
    };
    image.src = src;
  }

  function changeMask(file: File) {
    setMask(file);

    // always draw the mask to the canvas
    drawSource(file);
  }

  function changeSource(file: File) {
    setSource(file);

    // draw the source to the canvas if the mask has not been set
    if (doesExist(mask) === false) {
      drawSource(file);
    }
  }

  function grayscaleMask() {
    const canvas = mustExist(canvasRef.current);
    const ctx = mustExist(canvas.getContext('2d'));
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = image.data;

    for (let x = 0; x < canvas.width; ++x) {
      for (let y = 0; y < canvas.height; ++y) {
        const i = (y * canvas.width * PIXEL_SIZE) + (x * PIXEL_SIZE);
        const hue = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / PIXEL_WEIGHT;
        pixels[i] = hue;
        pixels[i + 1] = hue;
        pixels[i + 2] = hue;
      }
    }

    ctx.putImageData(image, 0, 0);
  }

  function floodMask(flooder: (n: number) => number) {
    const canvas = mustExist(canvasRef.current);
    const ctx = mustExist(canvas.getContext('2d'));
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = image.data;

    for (let x = 0; x < canvas.width; ++x) {
      for (let y = 0; y < canvas.height; ++y) {
        const i = (y * canvas.width * PIXEL_SIZE) + (x * PIXEL_SIZE);
        const hue = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / PIXEL_WEIGHT;
        const final = flooder(hue);

        pixels[i] = final;
        pixels[i + 1] = final;
        pixels[i + 2] = final;
      }
    }

    ctx.putImageData(image, 0, 0);
  }

  const upload = useMutation(uploadSource);
  const schedulers = useQuery('schedulers', async () => client.schedulers(), {
    staleTime: STALE_TIME,
  });

  // eslint-disable-next-line no-null/no-null
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [clicks, setClicks] = useState<Array<Point>>([]);

  const [painting, setPainting] = useState(false);
  const [brushColor, setBrushColor] = useState(0);
  const [brushSize, setBrushSize] = useState(DEFAULT_BRUSH);

  const [mask, setMask] = useState<File>();
  const [source, setSource] = useState<File>();
  const [params, setParams] = useState<BaseImgParams>({
    cfg: CONFIG_DEFAULTS.cfg.default,
    seed: CONFIG_DEFAULTS.seed.default,
    steps: CONFIG_DEFAULTS.steps.default,
    prompt: config.default.prompt,
  });
  const [scheduler, setScheduler] = useState(config.default.scheduler);

  useEffect(() => {
    const canvas = mustExist(canvasRef.current);
    const ctx = mustExist(canvas.getContext('2d'));
    ctx.fillStyle = grayToRGB(brushColor);

    for (const click of clicks) {
      ctx.beginPath();
      ctx.arc(click.x, click.y, brushSize, 0, FULL_CIRCLE);
      ctx.fill();
    }

    clicks.length = 0;
  }, [clicks.length]);

  function renderCanvas() {
    return <canvas
      ref={canvasRef}
      height={CONFIG_DEFAULTS.height.default}
      width={CONFIG_DEFAULTS.width.default}
      style={{
        maxHeight: CONFIG_DEFAULTS.height.default,
        maxWidth: CONFIG_DEFAULTS.width.default,
      }}
      onClick={(event) => {
        const canvas = mustExist(canvasRef.current);
        const bounds = canvas.getBoundingClientRect();

        setClicks([...clicks, {
          x: event.clientX - bounds.left,
          y: event.clientY - bounds.top,
        }]);
      }}
      onMouseDown={() => {
        setPainting(true);
      }}
      onMouseLeave={() => {
        setPainting(false);
      }}
      onMouseOut={() => {
        setPainting(false);
      }}
      onMouseUp={() => {
        setPainting(false);
      }}
      onMouseMove={(event) => {
        if (painting) {
          const canvas = mustExist(canvasRef.current);
          const bounds = canvas.getBoundingClientRect();

          setClicks([...clicks, {
            x: event.clientX - bounds.left,
            y: event.clientY - bounds.top,
          }]);
        }
      }}
    />;
  }

  return <Box>
    <Stack spacing={2}>
      <Stack direction='row' spacing={2}>
        <QueryList
          id='schedulers'
          labels={SCHEDULER_LABELS}
          name='Scheduler'
          result={schedulers}
          value={scheduler}
          onChange={(value) => {
            setScheduler(value);
          }}
        />
      </Stack>
      <ImageInput filter={IMAGE_FILTER} label='Source' onChange={changeSource} />
      <ImageInput filter={IMAGE_FILTER} label='Mask' onChange={changeMask} renderImage={renderCanvas} />
      <Stack direction='row' spacing={4}>
        <NumericField
          decimal
          label='Brush Shade'
          min={0}
          max={255}
          step={1}
          value={brushColor}
          onChange={(value) => {
            setBrushColor(value);
          }}
        />
        <NumericField
          decimal
          label='Brush Size'
          min={4}
          max={64}
          step={1}
          value={brushSize}
          onChange={(value) => {
            setBrushSize(value);
          }}
        />
        <Button
          startIcon={<FormatColorFill />}
          onClick={() => floodMask(floodBelow)}>
          Gray to black
        </Button>
        <Button
          startIcon={<Gradient />}
          onClick={() => grayscaleMask()}>
          Grayscale
        </Button>
        <Button
          startIcon={<FormatColorFill />}
          onClick={() => floodMask(floodAbove)}>
          Gray to white
        </Button>
      </Stack>
      <ImageControl params={params} onChange={(newParams) => {
        setParams(newParams);
      }} />
      <Button onClick={() => upload.mutate()}>Generate</Button>
      <MutationHistory result={upload} limit={4} element={ImageCard}
        isEqual={equalResponse}
      />
    </Stack>
  </Box>;
}
