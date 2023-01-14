import { doesExist, Maybe, mustExist } from '@apextoaster/js-utils';
import { FormatColorFill, Gradient } from '@mui/icons-material';
import { Button, Stack } from '@mui/material';
import { throttle } from 'lodash';
import React, { useEffect, useMemo, useRef, useState } from 'react';

import { ConfigParams, DEFAULT_BRUSH, SAVE_TIME } from '../config.js';
import { NumericField } from './NumericField';

export const FULL_CIRCLE = 2 * Math.PI;
export const MASK_OPACITY = 0.75;
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

export const MASK_STATE = {
  clean: 'clean',
  painting: 'painting',
  dirty: 'dirty',
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

  base?: Maybe<Blob>;
  source?: Maybe<Blob>;

  onSave: (blob: Blob) => void;
}

export function MaskCanvas(props: MaskCanvasProps) {
  const { base, config, source } = props;

  function floodMask(flood: FloodFn) {
    const buffer = mustExist(bufferRef.current);
    const ctx = mustExist(buffer.getContext('2d'));
    const image = ctx.getImageData(0, 0, buffer.width, buffer.height);
    const pixels = image.data;

    for (let x = 0; x < buffer.width; ++x) {
      for (let y = 0; y < buffer.height; ++y) {
        const i = (y * buffer.width * PIXEL_SIZE) + (x * PIXEL_SIZE);
        const hue = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / PIXEL_WEIGHT;
        const final = flood(hue);

        pixels[i] = final;
        pixels[i + 1] = final;
        pixels[i + 2] = final;
      }
    }

    ctx.putImageData(image, 0, 0);
    save();
  }

  function saveMask(): void {
    if (doesExist(bufferRef.current)) {
      if (maskState.current === MASK_STATE.clean) {
        return;
      }

      bufferRef.current.toBlob((blob) => {
        maskState.current = MASK_STATE.clean;
        props.onSave(mustExist(blob));
      });
    }
  }

  function drawBuffer() {
    if (doesExist(bufferRef.current) && doesExist(canvasRef.current)) {
      const dest = mustExist(canvasRef.current);
      const ctx = mustExist(dest.getContext('2d'));

      ctx.clearRect(0, 0, dest.width, dest.height);
      ctx.globalAlpha = MASK_OPACITY;
      ctx.drawImage(bufferRef.current, 0, 0);
    }
  }

  function drawCircle(ctx: CanvasRenderingContext2D, point: Point): void {
    ctx.beginPath();
    ctx.arc(point.x, point.y, brushSize, 0, FULL_CIRCLE);
    ctx.fill();
  }

  function drawSource(file: Blob): void {
    const image = new Image();
    image.onload = () => {
      const buffer = mustExist(bufferRef.current);
      const ctx = mustExist(buffer.getContext('2d'));
      ctx.drawImage(image, 0, 0);
      URL.revokeObjectURL(src);

      drawBuffer();
    };

    const src = URL.createObjectURL(file);
    image.src = src;
  }

  function finishPainting() {
    if (maskState.current === MASK_STATE.painting) {
      maskState.current = MASK_STATE.dirty;
      save();
    }
  }

  const save = useMemo(() => throttle(saveMask, SAVE_TIME), []);

  // eslint-disable-next-line no-null/no-null
  const bufferRef = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line no-null/no-null
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // painting state
  const maskState = useRef(MASK_STATE.clean);
  const [background, setBackground] = useState<string>();
  const [clicks, setClicks] = useState<Array<Point>>([]);
  const [brushColor, setBrushColor] = useState(DEFAULT_BRUSH.color);
  const [brushSize, setBrushSize] = useState(DEFAULT_BRUSH.size);

  useEffect(() => {
    // including clicks.length prevents the initial render from saving a blank canvas
    if (doesExist(bufferRef.current) && maskState.current === MASK_STATE.painting && clicks.length > 0) {
      const ctx = mustExist(bufferRef.current.getContext('2d'));
      ctx.fillStyle = grayToRGB(brushColor);

      for (const click of clicks) {
        drawCircle(ctx, click);
      }

      clicks.length = 0;
      drawBuffer();
    }
  }, [clicks.length]);

  useEffect(() => {
    if (maskState.current === MASK_STATE.dirty) {
      save();
    }
  }, [maskState.current]);

  useEffect(() => {
    if (doesExist(bufferRef.current) && doesExist(source)) {
      drawSource(source);
    }
  }, [source]);

  useEffect(() => {
    if (doesExist(base)) {
      if (doesExist(background)) {
        URL.revokeObjectURL(background);
      }

      setBackground(URL.createObjectURL(base));
    }
  }, [base]);

  const styles: React.CSSProperties = {
    maxHeight: config.height.default,
    maxWidth: config.width.default,
  };

  if (doesExist(background)) {
    styles.backgroundImage = `url(${background})`;
  }

  return <Stack spacing={2}>
    <canvas
      ref={bufferRef}
      height={config.height.default}
      width={config.width.default}
      style={{
        display: 'none',
      }}
    />
    <canvas
      ref={canvasRef}
      height={config.height.default}
      width={config.width.default}
      style={styles}
      onClick={(event) => {
        const canvas = mustExist(canvasRef.current);
        const bounds = canvas.getBoundingClientRect();

        const buffer = mustExist(bufferRef.current);
        const ctx = mustExist(buffer.getContext('2d'));
        ctx.fillStyle = grayToRGB(brushColor);

        drawCircle(ctx, {
          x: event.clientX - bounds.left,
          y: event.clientY - bounds.top,
        });

        maskState.current = MASK_STATE.dirty;
        save();
      }}
      onMouseDown={() => {
        maskState.current = MASK_STATE.painting;
      }}
      onMouseLeave={finishPainting}
      onMouseOut={finishPainting}
      onMouseUp={finishPainting}
      onMouseMove={(event) => {
        if (maskState.current === MASK_STATE.painting) {
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
