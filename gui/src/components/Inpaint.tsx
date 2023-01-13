import { doesExist, mustExist } from '@apextoaster/js-utils';
import { FormatColorFill, Gradient } from '@mui/icons-material';
import { Box, Button, Stack } from '@mui/material';
import { throttle } from 'lodash';
import * as React from 'react';
import { useCallback } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { useStore } from 'zustand';

import { ConfigParams, DEFAULT_BRUSH, IMAGE_FILTER, SAVE_TIME } from '../config.js';
import { ClientContext, StateContext } from '../state.js';
import { ImageControl } from './ImageControl.js';
import { ImageInput } from './ImageInput.js';
import { NumericField } from './NumericField.js';

const { useContext, useEffect, useRef, useState } = React;

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

export function floodGray(n: number): number {
  return n;
}

export function grayToRGB(n: number): string {
  return `rgb(${n.toFixed(0)},${n.toFixed(0)},${n.toFixed(0)})`;
}

export interface Point {
  x: number;
  y: number;
}

export interface InpaintProps {
  config: ConfigParams;

  model: string;
  platform: string;
}

export function Inpaint(props: InpaintProps) {
  const { config, model, platform } = props;
  const client = mustExist(useContext(ClientContext));

  function drawSource(file: Blob): Promise<void> {
    const image = new Image();
    return new Promise<void>((res, _rej) => {
      image.onload = () => {
        const canvas = mustExist(canvasRef.current);
        const ctx = mustExist(canvas.getContext('2d'));
        ctx.drawImage(image, 0, 0);
        URL.revokeObjectURL(src);

        // putting a save call here has a tendency to go into an infinite loop
        res();
      };

      const src = URL.createObjectURL(file);
      image.src = src;
    });
  }

  function saveMask(): Promise<void> {
    return new Promise((res, _rej) => {
      if (doesExist(canvasRef.current)) {
        canvasRef.current.toBlob((blob) => {
          setInpaint({
            mask: mustExist(blob),
          });
          res();
        });
      } else {
        res();
      }
    });
  }

  async function uploadSource(): Promise<void> {
    const output = await client.inpaint({
      ...params,
      model,
      platform,
      mask: mustExist(params.mask),
      source: mustExist(params.source),
    });

    setLoading(output);
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

    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    save();
  }

  function renderCanvas() {
    return <canvas
      ref={canvasRef}
      height={config.height.default}
      width={config.width.default}
      style={{
        maxHeight: config.height.default,
        maxWidth: config.width.default,
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

  const save = useCallback(throttle(saveMask, SAVE_TIME), []);
  const state = mustExist(useContext(StateContext));
  const params = useStore(state, (s) => s.inpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setInpaint = useStore(state, (s) => s.setInpaint);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setLoading = useStore(state, (s) => s.setLoading);

  const query = useQueryClient();
  const upload = useMutation(uploadSource, {
    onSuccess: () => query.invalidateQueries({ queryKey: 'ready' }),
  });
  // eslint-disable-next-line no-null/no-null
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // painting state
  const [clicks, setClicks] = useState<Array<Point>>([]);
  const [painting, setPainting] = useState(false);
  const [brushColor, setBrushColor] = useState(DEFAULT_BRUSH.color);
  const [brushSize, setBrushSize] = useState(DEFAULT_BRUSH.size);

  useEffect(function changeMask() {
    // always draw the new mask to the canvas
    if (doesExist(params.mask)) {
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      drawSource(params.mask);
    }
  }, [params.mask]);

  useEffect(function changeSource() {
    // draw the source to the canvas if the mask has not been set
    if (doesExist(params.source) && doesExist(params.mask) === false) {
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      drawSource(params.source);
    }
  }, [params.source]);

  useEffect(() => {
    // including clicks.length prevents the initial render from saving a blank canvas
    if (doesExist(canvasRef.current) && clicks.length > 0) {
      const ctx = mustExist(canvasRef.current.getContext('2d'));
      ctx.fillStyle = grayToRGB(brushColor);

      for (const click of clicks) {
        ctx.beginPath();
        ctx.arc(click.x, click.y, brushSize, 0, FULL_CIRCLE);
        ctx.fill();
      }

      clicks.length = 0;

      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      save();
    }
  }, [clicks.length]);

  return <Box>
    <Stack spacing={2}>
      <ImageInput filter={IMAGE_FILTER} image={params.source} label='Source' onChange={(file) => {
        setInpaint({
          source: file,
        });
      }} />
      <ImageInput filter={IMAGE_FILTER} image={params.mask} label='Mask' onChange={(file) => {
        setInpaint({
          mask: file,
        });
      }} renderImage={renderCanvas} />
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
          variant='outlined'
          startIcon={<FormatColorFill />}
          onClick={() => floodMask(floodBelow)}>
          Gray to black
        </Button>
        <Button
          variant='outlined'
          startIcon={<Gradient />}
          onClick={() => floodMask(floodGray)}>
          Grayscale
        </Button>
        <Button
          variant='outlined'
          startIcon={<FormatColorFill />}
          onClick={() => floodMask(floodAbove)}>
          Gray to white
        </Button>
      </Stack>
      <ImageControl
        config={config}
        params={params}
        onChange={(newParams) => {
          setInpaint(newParams);
        }}
      />
      <Button onClick={() => upload.mutate()}>Generate</Button>
    </Stack>
  </Box>;
}
