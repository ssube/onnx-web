import { doesExist, Maybe, mustExist } from '@apextoaster/js-utils';
import { FormatColorFill, Gradient } from '@mui/icons-material';
import { Button, Stack } from '@mui/material';
import { throttle } from 'lodash';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { ConfigParams, DEFAULT_BRUSH, SAVE_TIME } from '../config.js';
import { NumericField } from './NumericField';

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

export type FloodFn = (n: number) => number;

export interface Point {
  x: number;
  y: number;
}

export interface MaskCanvasProps {
  config: ConfigParams;

  source?: Maybe<Blob>;

  onSave: (blob: Blob) => void;
}

export function MaskCanvas(props: MaskCanvasProps) {
  const { config, source } = props;

  function floodMask(flood: FloodFn) {
    const canvas = mustExist(canvasRef.current);
    const ctx = mustExist(canvas.getContext('2d'));
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = image.data;

    for (let x = 0; x < canvas.width; ++x) {
      for (let y = 0; y < canvas.height; ++y) {
        const i = (y * canvas.width * PIXEL_SIZE) + (x * PIXEL_SIZE);
        const hue = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / PIXEL_WEIGHT;
        const final = flood(hue);

        pixels[i] = final;
        pixels[i + 1] = final;
        pixels[i + 2] = final;
      }
    }

    ctx.putImageData(image, 0, 0);

    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    save();
  }

  function saveMask(): Promise<void> {
    // eslint-disable-next-line no-console
    console.log('starting canvas save');

    return new Promise((res, _rej) => {
      if (doesExist(canvasRef.current)) {
        canvasRef.current.toBlob((blob) => {
          // eslint-disable-next-line no-console
          console.log('finishing canvas save');

          props.onSave(mustExist(blob));
          res();
        });
      } else {
        res();
      }
    });
  }

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

  const save = useMemo(() => throttle(saveMask, SAVE_TIME), []);

  // eslint-disable-next-line no-null/no-null
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // painting state
  const [clicks, setClicks] = useState<Array<Point>>([]);
  const [painting, setPainting] = useState(false);
  const [brushColor, setBrushColor] = useState(DEFAULT_BRUSH.color);
  const [brushSize, setBrushSize] = useState(DEFAULT_BRUSH.size);

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
    }
  }, [clicks.length]);

  useEffect(() => {
    if (painting === false) {
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      save();
    }
  }, [painting]);

  useEffect(() => {
    if (doesExist(canvasRef.current) && doesExist(source)) {
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      drawSource(source);
    }
  }, [source]);

  return <Stack spacing={2}>
    <canvas
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
    />
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
    </Stack></Stack>;
}
