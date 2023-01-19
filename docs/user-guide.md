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
  - [Errors](#errors)
    - [Image Errors](#image-errors)
      - [Empty black images](#empty-black-images)
      - [Distorted and noisy images](#distorted-and-noisy-images)
      - [Scattered image tiles](#scattered-image-tiles)
    - [Client Errors](#client-errors)
      - [Error fetching server parameters](#error-fetching-server-parameters)
      - [Parameter version error](#parameter-version-error)
    - [Server Errors](#server-errors)
      - [Very slow with high CPU usage, max fan speed during image generation](#very-slow-with-high-cpu-usage-max-fan-speed-during-image-generation)
      - [ONNXRuntimeError: The parameter is incorrect](#onnxruntimeerror-the-parameter-is-incorrect)
      - [The expanded size of the tensor must match the existing size](#the-expanded-size-of-the-tensor-must-match-the-existing-size)

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

### Modes and tabs

- [txt2img](#txt2img-tab)
  - generate a new image from a text prompt
- [img2img](#img2img-tab)
  - modify an existing image using a text prompt
- [inpaint](#inpaint-tab)
  - modify parts of an existing image using an opacity mask
  - includes [outpaint](#outpaint-parameters)
- [upscale](#upscale-tab)
  - resize an existing image

### Common parameters

These are common parameters shared by the diffusion models and all tabs that use diffusers (txt2img, img2img, and
inpaint).

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

The input text for your image, things that should be included.

#### Negative prompt parameter

The opposite of [the prompt parameter](#prompt-parameter), things that should _not_ be included.

### Upscaling parameters

Resize the output image before returning it to the client.

Enabling this will run Real ESRGAN and requires an upscaling model.

#### Scale parameter

The output scale for Real ESRGAN.

#### Outscale parameter

The final output scale after running Real ESRGAN. This can increase _or_ decrease the size of the final
output. Lanczos interpolation is used when the outscale is greater than the scale.

#### Denoise parameter

#### Face correction and strength

Run face correction the the output image before returning it to the client.

Enabling this will run GFPGAN and requires a correction model.

### Scheduler comparison

https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers#compare-schedulers

## Tabs

### Txt2img tab

The txt2img tab turns your wildest ideas into something resembling them, maybe.

This mode takes a text prompt along with various other parameters and produces a new image.

#### Width and height parameters

Controls the size of the output image, before upscaling.

### Img2img tab

The img2img tab takes a source image along with the text prompt and produces a similar image. You can use the
strength parameter to control the level of similarity between the source and output.

The output image will be the same size as the input, unless upscaling is turned on.

#### Img2img source image

Upload a source image for img2img.

### Inpaint tab

The inpaint tab provides a way to edit part of an image and run the diffusion pipeline again, without editing
the entire image. It still takes a text prompt, but uses a mask to decide which pixels should be regenerated.

The mask can be uploaded or edited directly in the browser. White pixels in the mask will be replaced with pixels
from the noise source, then replaced again by the diffusion pipeline. Black pixels in the mask will be kept as
they appeared in the source. The mask can use gray values to blend the difference.

When all of the options are used together, the process is:

1. Add borders to the source image
2. Generate the noise source from the source image and random data
3. Run the mask filter on the mask image
4. Blend the source image and the noise source using the mask image (pre-multiply)
5. Apply the diffusion model to the source image using the mask image to weight pixels
6. Apply the upscaling and correction models to the output
7. Save the output

#### Inpaint source image

Upload a source image for inpaint.

#### Mask canvas and brush parameters

Upload or draw a mask image.

White pixels will be replaced with noise and then regenerated, black pixels will be kept as-is in the output.

- Gray to black
  - Convert gray parts of the mask to black (keep them)
- Fill with black
  - Keep all pixels
- Fill with white
  - Replace all pixels
- Gray to white
  - Convert gray parts of the mask to white (replace them)

#### Mask filter parameter

Mask filters are used to pre-process the mask before blending the source image with the noise and before running
the diffusion pipeline.

- None
  - no mask filter
  - usually a fine option
- Gaussian Multiply
  - blur and darken the mask
  - good when you want to soften the edges of the area to be kept, without shrinking it
- Gaussian Screen
  - blur and lighten the mask
  - good when you want to soften the edges of the area to be kept and do not mind shrinking it slightly

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

## Errors

### Image Errors

#### Empty black images

#### Distorted and noisy images

#### Scattered image tiles

### Client Errors

#### Error fetching server parameters

This can happen when the client cannot fetch the server parameters because the request times out or has been rejected
by the server.

This often means that the requested API server is not running.

#### Parameter version error

This can happen when the version in the server parameters is too old for the current client or missing entirely, which
was the case before version v0.5.0.

This often means that the API server is running but out-of-date.

### Server Errors

If your image fails to render without any other error messages on the client, check the server logs for errors (if you
have access).

#### Very slow with high CPU usage, max fan speed during image generation

This can happen when you attempt to use a platform that is not supported by the current hardware.

This often means that you need to select a different platform or install the correct drivers for your GPU and operating
system.

Example error:

```none
loading different pipeline
C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:54: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'DmlExecutionProvider, CPUExecutionProvider'
```

The `CPUExecutionProvider` is used as a fallback, but has a tendency to max out all of your real CPU cores.

#### ONNXRuntimeError: The parameter is incorrect

This can happen when you attempt to use an inpainting model with txt2img or img2img mode, or a regular model for inpaint
mode.

This often means that you are using an invalid model for the current tab.

Example error:

```none
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_web\pipeline.py", line 181, in run_inpaint_pipeline
    image = pipe(
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\torch\autograd\grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\diffusers\pipelines\stable_diffusion\pipeline_onnx_stable_diffusion_inpaint.py", line 427, in __call__
    noise_pred = self.unet(
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\diffusers\onnx_utils.py", line 61, in __call__
    return self.model.run(None, inputs)
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 200, in run
    return self._sess.run(output_names, input_feed, run_options)
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Conv node. Name:'/conv_in/Conv' Status Message: D:\a\_work\1\s\onnx
runtime\core\providers\dml\DmlExecutionProvider\src\MLOperatorAuthorImpl.cpp(1878)\onnxruntime_pybind11_state.pyd!00007FFB8404F72D: (caller: 00007FFB84050AEF) Exception(15) tid(2428) 80070057 The parameter is incorrect
```

#### The expanded size of the tensor must match the existing size

This can happen when you use an upscaling model that was trained at one specific scale with a different scale that
it was not expecting.

This often means that you are using an invalid scale for the upscaling model you have selected.

Example error:

```none
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_web\upscale.py", line 155, in upscale_resrgan
    output, _ = upsampler.enhance(output, outscale=params.outscale)
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\torch\autograd\grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\realesrgan\utils.py", line 228, in enhance
    self.tile_process()
  File "C:\Users\ssube\stabdiff\onnx-web\api\onnx_env\lib\site-packages\realesrgan\utils.py", line 182, in tile_process
    self.output[:, :, output_start_y:output_end_y,
RuntimeError: The expanded size of the tensor (2048) must match the existing size (1024) at non-singleton dimension 3.  Target sizes: [1, 3, 2048, 2048].  Tensor sizes: [3, 1024, 1024]
```
