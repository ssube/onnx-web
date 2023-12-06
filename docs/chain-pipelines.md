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
      - [Blend: Img2img](#blend-img2img)
      - [Blend: Inpaint](#blend-inpaint)
    - [Correction Stages](#correction-stages)
      - [Correct: CodeFormer](#correct-codeformer)
      - [Correct: GFPGAN](#correct-gfpgan)
    - [Persistence Stages](#persistence-stages)
      - [Persist: Disk](#persist-disk)
      - [Persist: S3](#persist-s3)
    - [Reduction Stages](#reduction-stages)
      - [Reduce: Crop](#reduce-crop)
      - [Reduce: Thumbnail](#reduce-thumbnail)
    - [Source Stages](#source-stages)
      - [Source: Noise](#source-noise)
      - [Source: Txt2img](#source-txt2img)
    - [Upscaling Stages](#upscaling-stages)
      - [Upscale: Outpaint](#upscale-outpaint)
      - [Upscale: Real ESRGAN](#upscale-real-esrgan)
      - [Upscale: Stable Diffusion](#upscale-stable-diffusion)

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

#### Blend: Img2img

#### Blend: Inpaint

### Correction Stages

#### Correct: CodeFormer

#### Correct: GFPGAN

### Persistence Stages

#### Persist: Disk

#### Persist: S3

### Reduction Stages

#### Reduce: Crop

#### Reduce: Thumbnail

### Source Stages

#### Source: Noise

#### Source: Txt2img

### Upscaling Stages

#### Upscale: Outpaint

Upscaling stage using outpainting. This adds empty borders to the source image, optionally fills them with noise, and
then runs inpainting on those areas.

#### Upscale: Real ESRGAN

Upscaling stage using the Real ESRGAN upscaling models, available in x2 and x4 versions:

- https://github.com/xinntao/Real-ESRGAN/releases

#### Upscale: Stable Diffusion

Upscaling stage using the Stable Diffusion x4 upscaling model:

- https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
- https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx
