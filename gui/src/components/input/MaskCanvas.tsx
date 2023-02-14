import { doesExist, Maybe, mustExist } from '@apextoaster/js-utils';
import { Download, FormatColorFill, Gradient, InvertColors, Save, Undo } from '@mui/icons-material';
import { Button, Stack, Typography } from '@mui/material';
import { throttle } from 'lodash';
import React, { RefObject, useContext, useEffect, useMemo, useRef } from 'react';
import { useStore } from 'zustand';

import { SAVE_TIME } from '../../config.js';
import { ConfigContext, LoggerContext, StateContext } from '../../state.js';
import { imageFromBlob } from '../../utils.js';
import { NumericField } from './NumericField';

export const DRAW_TIME = 25;
export const FULL_CIRCLE = 2 * Math.PI;
export const FULL_OPACITY = 1.0;
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

export type FloodFn = (n: number) => number;

export interface Point {
  x: number;
  y: number;
}

export interface MaskCanvasProps {
  source?: Maybe<Blob>;
  mask?: Maybe<Blob>;

  onSave: (blob: Blob) => void;
}

export function MaskCanvas(props: MaskCanvasProps) {
  const { source, mask } = props;
  const { params } = mustExist(useContext(ConfigContext));
  const logger = mustExist(useContext(LoggerContext));

  function composite() {
    if (doesExist(viewRef.current)) {
      const { ctx } = getClearContext(viewRef);
      ctx.globalAlpha = MASK_OPACITY;

      if (doesExist(maskRef.current)) {
        ctx.drawImage(maskRef.current, 0, 0);
      }

      if (doesExist(brushRef.current) && painting.current === false) {
        ctx.drawImage(brushRef.current, 0, 0);
      }
    }
  }

  function drawBrush(point: Point): void {
    const { ctx } = getClearContext(brushRef);
    ctx.fillStyle = grayToRGB(brush.color, brush.strength);

    drawCircle(ctx, {
      x: point.x,
      y: point.y,
    }, brush.size);

    composite();
  }

  function drawClicks(clicks: Array<Point>): void {
    if (clicks.length > 0) {
      logger.debug('drawing clicks', { count: clicks.length });

      const { ctx } = getContext(maskRef);
      ctx.fillStyle = grayToRGB(brush.color, brush.strength);

      for (const click of clicks) {
        drawCircle(ctx, click, brush.size);
      }

      composite();
      dirty.current = true;
    }
  }

  async function drawMask(file: Blob): Promise<void> {
    const image = await imageFromBlob(file);
    if (doesExist(maskRef.current)) {
      logger.debug('draw mask');

      const { canvas, ctx } = getClearContext(viewRef);
      ctx.globalAlpha = FULL_OPACITY;
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

      // if mask was recently changed, it doesn't need to be saved - avoid loops
      dirty.current = false;
      composite();
    }
  }

  function drawMouse(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = mustExist(viewRef.current);
    const bounds = canvas.getBoundingClientRect();

    if (painting.current) {
      drawClicks([{
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      }]);
    } else {
      drawBrush({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });
    }
  }

  function drawUndo(): void {
    if (doesExist(maskRef.current) && doesExist(undoRef.current)) {
      logger.debug('draw undo');

      const { ctx } = getClearContext(maskRef);
      ctx.drawImage(undoRef.current, 0, 0);

      composite();
      save();
    }
  }

  function finishPainting() {
    logger.debug('finish painting');
    painting.current = false;

    if (doesExist(brushRef.current)) {
      getClearContext(brushRef);
    }

    if (dirty.current) {
      save();
    }
  }

  function drawFill(fn: FloodFn): void {
    saveUndo();
    floodCanvas(maskRef, fn);
    composite();
    dirty.current = true;
  }

  /**
   * Save the current mask to the undo canvas.
   */
  function saveUndo(): void {
    if (doesExist(maskRef.current) && doesExist(undoRef.current)) {
      logger.debug('save undo');
      const { ctx } = getClearContext(undoRef);
      ctx.drawImage(maskRef.current, 0, 0);
    }
  }

  /**
   * Save the current mask to state, so that it persists between tabs.
   */
  function saveMask(): void {
    if (doesExist(maskRef.current)) {
      logger.debug('save mask', { dirty: dirty.current });
      if (dirty.current === false) {
        return;
      }

      maskRef.current.toBlob((blob) => {
        dirty.current = false;
        props.onSave(mustExist(blob));
      });
    }
  }

  const save = useMemo(() => throttle(saveMask, SAVE_TIME), []);

  // eslint-disable-next-line no-null/no-null
  const brushRef = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line no-null/no-null
  const maskRef = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line no-null/no-null
  const viewRef = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line no-null/no-null
  const undoRef = useRef<HTMLCanvasElement>(null);

  // painting state
  const painting = useRef(false);
  const dirty = useRef(false);
  const background = useRef<string>();

  const state = mustExist(useContext(StateContext));
  const brush = useStore(state, (s) => s.brush);
  // eslint-disable-next-line @typescript-eslint/unbound-method
  const setBrush = useStore(state, (s) => s.setBrush);

  useEffect(() => {
    if (dirty.current) {
      save();
    }
  }, [dirty.current]);

  useEffect(() => {
    if (doesExist(maskRef.current) && doesExist(mask)) {
      drawMask(mask).catch((err) => {
        logger.error(err, 'error drawing mask for effect');
      });
    }
  }, [mask]);

  useEffect(() => {
    if (doesExist(source)) {
      if (doesExist(background.current)) {
        URL.revokeObjectURL(background.current);
      }

      background.current = URL.createObjectURL(source);

      // initialize the mask if it does not exist
      if (doesExist(mask) === false) {
        getClearContext(maskRef);
        dirty.current = true;
      }
    }
  }, [source]);

  const backgroundStyle: React.CSSProperties = {
    backgroundPosition: 'top left',
    backgroundRepeat: 'no-repeat',
    backgroundSize: 'contain',
    border: '1px solid black',
    maxHeight: params.height.default,
    maxWidth: params.width.default,
  };

  if (doesExist(background.current)) {
    backgroundStyle.backgroundImage = `url(${background.current})`;
  }

  const hiddenStyle: React.CSSProperties = {
    ...backgroundStyle,
    display: 'none',
  };

  return <Stack spacing={2}>
    <Stack direction='row' spacing={2}>
      <Button
        variant='outlined'
        startIcon={<Download />}
        onClick={() => {
          if (doesExist(maskRef.current)) {
            const data = maskRef.current.toDataURL('image/png');
            const link = document.createElement('a');

            link.setAttribute('download', 'mask.png');
            link.setAttribute('href', data.replace('image/png', 'image/octet-stream'));
            link.click();
          }
        }}
      />
      <Button
        variant='outlined'
        startIcon={<Undo />}
        onClick={() => drawUndo()}
      />
    </Stack>
    <canvas
      ref={brushRef}
      height={params.height.default}
      width={params.width.default}
      style={hiddenStyle}
    />
    <canvas
      ref={maskRef}
      height={params.height.default}
      width={params.width.default}
      style={hiddenStyle}
    />
    <canvas
      ref={undoRef}
      height={params.height.default}
      width={params.width.default}
      style={hiddenStyle}
    />
    <canvas
      ref={viewRef}
      height={params.height.default}
      width={params.width.default}
      style={backgroundStyle}
      onMouseDown={(event) => {
        logger.debug('mouse down', { state: painting.current });

        saveUndo();
        painting.current = true;

        drawMouse(event);
      }}
      onMouseLeave={finishPainting}
      onMouseOut={finishPainting}
      onMouseUp={finishPainting}
      onMouseMove={drawMouse}
    />
    <Typography variant='body1'>
      Black pixels in the mask will stay the same, white pixels will be replaced. The masked pixels will be blended
      with the noise source before the diffusion model runs, giving it more variety to use.
    </Typography>
    <Stack>
      <Stack direction='row' spacing={4}>
        <NumericField
          label='Brush Color'
          min={COLORS.black}
          max={COLORS.white}
          step={1}
          value={brush.color}
          onChange={(color) => {
            setBrush({ color });
          }}
        />
        <NumericField
          label='Brush Size'
          min={1}
          max={64}
          step={1}
          value={brush.size}
          onChange={(size) => {
            setBrush({ size });
          }}
        />
        <NumericField
          decimal
          label='Brush Strength'
          min={0}
          max={1}
          step={0.01}
          value={brush.strength}
          onChange={(strength) => {
            setBrush({ strength });
          }}
        />
      </Stack>
      <Stack direction='row' spacing={2}>
        <Button
          variant='outlined'
          startIcon={<FormatColorFill />}
          onClick={() => {
            drawFill(floodBlack);
          }}>
          Fill with black
        </Button>
        <Button
          variant='outlined'
          startIcon={<FormatColorFill />}
          onClick={() => {
            drawFill(floodWhite);
          }}>
          Fill with white
        </Button>
        <Button
          variant='outlined'
          startIcon={<InvertColors />}
          onClick={() => {
            drawFill(floodInvert);
          }}>
          Invert
        </Button>
        <Button
          variant='outlined'
          startIcon={<Gradient />}
          onClick={() => {
            drawFill(floodBelow);
          }}>
          Gray to black
        </Button>
        <Button
          variant='outlined'
          startIcon={<Gradient />}
          onClick={() => {
            drawFill(floodAbove);
          }}>
          Gray to white
        </Button>
      </Stack>
    </Stack>
  </Stack>;
}

function getContext(ref: RefObject<HTMLCanvasElement>) {
  const canvas = mustExist(ref.current);
  const ctx = mustExist(canvas.getContext('2d'));

  return { canvas, ctx };
}

function getClearContext(ref: RefObject<HTMLCanvasElement>) {
  const ret = getContext(ref);
  ret.ctx.clearRect(0, 0, ret.canvas.width, ret.canvas.height);

  return ret;
}

function drawCircle(ctx: CanvasRenderingContext2D, point: Point, size: number): void {
  ctx.beginPath();
  ctx.arc(point.x, point.y, size, 0, FULL_CIRCLE);
  ctx.fill();
}

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

export function floodBlack(): number {
  return COLORS.black;
}

export function floodWhite(): number {
  return COLORS.white;
}

export function floodInvert(n: number): number {
  return COLORS.white - n;
}

export function grayToRGB(n: number, o = 1.0): string {
  return `rgba(${n.toFixed(0)},${n.toFixed(0)},${n.toFixed(0)},${o.toFixed(2)})`;
}

function floodCanvas(ref: RefObject<HTMLCanvasElement>, flood: FloodFn) {
  const { canvas, ctx } = getContext(ref);
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
      // eslint-disable-next-line @typescript-eslint/no-magic-numbers
      pixels[i + 3] = COLORS.white; // fully opaque
    }
  }

  ctx.putImageData(image, 0, 0);
}
