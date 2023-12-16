# Chain Pipelines

Chain pipelines are a new feature in v0.6 that allows you to run any combination of models on images
of almost any size, by automatically splitting them into smaller tiles as needed. Individual models
are run on each tile, then the results are recombined and passed on to the next stage.

## Contents

- [Chain Pipelines](#chain-pipelines)
  - [Contents](#contents)
  - [Overview](#overview)
    - [Format](#format)
  - [Stages](#stages)
    - [Blending Stages](#blending-stages)
      - [Blend: Denoise](#blend-denoise)
      - [Blend: Grid](#blend-grid)
      - [Blend: Img2img](#blend-img2img)
      - [Blend: Linear](#blend-linear)
      - [Blend: Mask](#blend-mask)
    - [Correction Stages](#correction-stages)
      - [Correct: CodeFormer](#correct-codeformer)
      - [Correct: GFPGAN](#correct-gfpgan)
    - [Compound Stages](#compound-stages)
      - [Highres Stage](#highres-stage)
      - [Upscale Stage](#upscale-stage)
    - [Persistence Stages](#persistence-stages)
      - [Persist: Disk](#persist-disk)
      - [Persist: S3](#persist-s3)
    - [Reduction Stages](#reduction-stages)
      - [Reduce: Crop](#reduce-crop)
      - [Reduce: Thumbnail](#reduce-thumbnail)
    - [Source Stages](#source-stages)
      - [Source: Noise](#source-noise)
      - [Source: S3](#source-s3)
      - [Source: Txt2img](#source-txt2img)
      - [Source: URL](#source-url)
    - [Upscaling Stages](#upscaling-stages)
      - [Upscale: BSRGAN](#upscale-bsrgan)
      - [Upscale: Highres](#upscale-highres)
      - [Upscale: Outpaint](#upscale-outpaint)
      - [Upscale: Real ESRGAN](#upscale-real-esrgan)
      - [Upscale: Simple](#upscale-simple)
      - [Upscale: Stable Diffusion](#upscale-stable-diffusion)
      - [Upscale: SwinIR](#upscale-swinir)

## Overview

### Format

The `/api/chain` endpoint accepts a chain pipeline in JSON format and adds it to the queue of background jobs.

Pipelines are defined mostly through their `stages`, where each stage specifies a function to be run and the
parameters for that function, including the name of the model to be used.

The output of the pipeline _will not_ automatically be saved to disk, which is the case for the single-stage
endpoints. You must use at least one `persist-*` stage. Persist stages can be placed anywhere in the pipeline
and can also save intermediate output, such as the result of a `source-txt2img` stage before upscaling it.

```json
{
  "stages": [
    {
      "name": "start",
      "type": "source-txt2img",
      "params": {
        "prompt": "a magical wizard"
      }
    },
    {
      "name": "expand",
      "type": "upscale-outpaint",
      "params": {
        "border": 256,
        "prompt": "a magical wizard in a robe fighting a dragon"
      }
    },
    {
      "name": "save-local",
      "type": "persist-disk",
      "params": {
        "tiles": "hd8k"
      }
    }
  ]
}
```

The complete schema can be found in [`api/schema.yaml`](../api/schema.yaml) and some example pipelines are available
in [`common/pipelines`](../common/pipelines).

## Stages

### Blending Stages

#### Blend: Denoise

Run [fast non-local means denoising](https://docs.opencv.org/4.8.0/d5/d69/tutorial_py_non_local_means.html) using `cv2`.

#### Blend: Grid

Combine the source images into a grid.

#### Blend: Img2img

Run an img2img pipeline.

#### Blend: Linear

Blend two images using linear interpolation (0.0 is the first image, 1.0 is the second).

#### Blend: Mask

Blend two images using a mask.

### Correction Stages

#### Correct: CodeFormer

Run correction using CodeFormer.

#### Correct: GFPGAN

Run correction using GFPGAN.

### Compound Stages

Not currently available through JSON API.

#### Highres Stage

TODO

#### Upscale Stage

TODO

### Persistence Stages

#### Persist: Disk

Save all of the sources to disk.

#### Persist: S3

Save all of the sources to an S3 bucket.

### Reduction Stages

#### Reduce: Crop

TODO

#### Reduce: Thumbnail

TODO

### Source Stages

#### Source: Noise

Create a new source using a noise generator.

#### Source: S3

Load a new source from an S3 bucket.

#### Source: Txt2img

Run a txt2img pipeline.

#### Source: URL

Load a new source from a URL.

### Upscaling Stages

#### Upscale: BSRGAN

Upscaling stage using BSRGAN.

#### Upscale: Highres

Upscaling stage using highres.

#### Upscale: Outpaint

Upscaling stage using outpainting. This adds empty borders to the source image, optionally fills them with noise, and
then runs inpainting on those areas.

#### Upscale: Real ESRGAN

Upscaling stage using the Real ESRGAN upscaling models, available in x2 and x4 versions:

- https://github.com/xinntao/Real-ESRGAN/releases

#### Upscale: Simple

Upscaling stage using bilinear or Lanczos upscaling.

#### Upscale: Stable Diffusion

Upscaling stage using the Stable Diffusion x4 upscaling model:

- https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
- https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx

#### Upscale: SwinIR

Upscaling stage using SwinIR.
