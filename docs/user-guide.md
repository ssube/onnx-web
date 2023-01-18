# User Guide

This is the user guide for ONNX web, a web GUI for running hardware-accelerated ONNX models.

## Contents

- [User Guide](#user-guide)
  - [Contents](#contents)
  - [Outline](#outline)
    - [ONNX Models](#onnx-models)
    - [Modes and tabs](#modes-and-tabs)
    - [Common parameters](#common-parameters)
      - [Scheduler parameter](#scheduler-parameter)
      - [CFG parameter](#cfg-parameter)
      - [Steps parameter](#steps-parameter)
      - [Seed parameter](#seed-parameter)
      - [Prompt parameter](#prompt-parameter)
      - [Negative prompt parameter](#negative-prompt-parameter)
    - [Upscaling parameters](#upscaling-parameters)
      - [Scale parameter](#scale-parameter)
      - [Outscale parameter](#outscale-parameter)
      - [Denoise parameter](#denoise-parameter)
      - [Face correction and strength](#face-correction-and-strength)
    - [Scheduler comparison](#scheduler-comparison)
  - [Tabs](#tabs)
    - [Txt2img tab](#txt2img-tab)
      - [Width and height parameters](#width-and-height-parameters)
    - [Img2img tab](#img2img-tab)
      - [Img2img source image](#img2img-source-image)
    - [Inpaint tab](#inpaint-tab)
      - [Inpaint source image](#inpaint-source-image)
      - [Mask canvas and brush parameters](#mask-canvas-and-brush-parameters)
      - [Mask filter parameter](#mask-filter-parameter)
      - [Noise source parameter](#noise-source-parameter)
      - [Outpaint parameters](#outpaint-parameters)
    - [Upscale tab](#upscale-tab)
    - [Settings tab](#settings-tab)
    - [Error screen](#error-screen)
      - [Server params error](#server-params-error)
      - [Params version error](#params-version-error)

## Outline

### ONNX Models

Models are split up into three groups:

1. Diffusion
   1. Stable Diffusion
2. Upscaling
   1. Real ESRGAN
3. Correction
   1. GFPGAN

There are many models, variations, and versions available.

TODO: describe some and provide links

### Modes and tabs

- txt2img
- img2img
- inpaint
  - outpaint
- upscale

### Common parameters

Using the same prompt and seed should produce similar images. Using the same prompt, seed, steps, and CFG should
produce the same image.

#### Scheduler parameter

#### CFG parameter

Classifier free guidance.

#### Steps parameter

The number of scheduler steps to run.

#### Seed parameter

The random seed.

#### Prompt parameter

The input text for your image.

#### Negative prompt parameter

### Upscaling parameters

#### Scale parameter

#### Outscale parameter

#### Denoise parameter

#### Face correction and strength

### Scheduler comparison

## Tabs

### Txt2img tab

The txt2img tab turns your wildest ideas into something resembling them, maybe.

This mode takes a text prompt along with various other parameters and produces a new image.

#### Width and height parameters

Controls the size of the output image, before upscaling.

### Img2img tab

The img2img tab takes a source image along with the text prompt and produces a similar image. You can use the
strength parameter to control the level of similarity between the source and output.

The output image will be the same size as the input, before upscaling.

#### Img2img source image

Upload a source image.

### Inpaint tab

The inpaint tab provides a way to edit part of an image and run the diffusion pipeline again, without editing
the entire image. It still takes a text prompt, but uses a mask to decide which pixels should be regenerated.

The mask can be uploaded or edited directly in the browser. White pixels in the mask will be replaced with pixels
from the noise source, then replaced again by the diffusion pipeline. Black pixels in the mask will be kept as
they appeared in the source. The mask can use gray values to blend the difference.

When all of the options are used together, the process is:

1. Expand the source image, centering the existing pixels
2. Generate noise source from source image and random data
3. Run the mask filter on the mask image
4. Blend the source image and noise source using the mask image
5. Apply the diffusion model using the mask image
6. Apply the upscaling and correction models
7. Save the output

#### Inpaint source image

Upload a source image.

#### Mask canvas and brush parameters

Upload or draw a mask.

White pixels will be replaced, black pixels will be kept from the source.

- Gray to black
  - Convert gray parts of the mask to black (keep)
- Fill with black
  - Keep all pixels
- Fill with white
  - Replace all pixels
- Gray to white
  - Convert gray parts of the mask to white (replace)

#### Mask filter parameter

Mask filters are used to pre-process the mask before blending the source image with the noise and before running
the diffusion pipeline.

#### Noise source parameter

Noise sources are used to create new data for the next round of diffusion. Sometimes adding noise can improve
the results, but it may also be too much. A variety of sources are provided.

- Fill Edges
  - fill the edges of the image with a solid color
  - only changes the image when used with outpainting
- Fill Masked
  - fills the edges and masked areas of the image with a solid color
- Gaussian Blur
  - blur the source image
  - fills the edges with noise when used with outpainting
  - a good option for finishing the edges of an image
- Histogram Noise
  - fills the edges and masked area with noise matching the source color palette
  - noise color is based on the color frequency in the source histogram
  - a good option for continuing to build an image
- Gaussian Noise
  - fills the edges and masked area with Gaussian noise
- Uniform Noise
  - fills the edges and masked area with uniform noise

#### Outpaint parameters

The number of pixels to add in each direction.

### Upscale tab

The upscale tab provides a dedicated way to upscale an image and run face correction using Real ESRGAN and GFPGAN,
without running a diffusion pipeline at all. This can be faster and avoids making unnecessary changes to the image.

### Settings tab

The settings tab provides access to some of the settings and allows you to reset the state of the other tabs
to the defaults, if they get out of control.

Changing the API server will reload the client.

### Error screen

#### Server params error

This happens when the client cannot fetch the server parameters.

#### Params version error

This happens when the version in the server parameters is too old for the client or does not exist.
