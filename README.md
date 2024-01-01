# onnx-web

onnx-web is designed to simplify the process of running Stable Diffusion and other [ONNX models](https://onnx.ai) so you
can focus on making high quality, high resolution art. With the efficiency of hardware acceleration on both AMD and
Nvidia GPUs, and offering a reliable CPU software fallback, it offers the full feature set on desktop, laptops, and
multi-GPU servers with a seamless user experience.

You can navigate through the user-friendly web UI, hosted on Github Pages and accessible across all major browsers,
including your go-to mobile device. Here, you have the flexibility to choose diffusion models and accelerators for each
image pipeline, with easy access to the image parameters that define each modes. Whether you're uploading images or
expressing your artistic touch through inpainting and outpainting, onnx-web provides an environment that's as
user-friendly as it is powerful. Recent output images are neatly presented beneath the controls, serving as a handy
visual reference to revisit previous parameters or remix your earlier outputs.

Dive deeper into the onnx-web experience with its API, compatible with both Linux and Windows. This RESTful interface
seamlessly integrates various pipelines from the [HuggingFace diffusers](https://huggingface.co/diffusers/main/en/index)
library, offering valuable metadata on models and accelerators, along with detailed outputs from your creative runs.

Embark on your generative art journey with onnx-web, and explore its capabilities through our detailed documentation
site. Find a comprehensive getting started guide, setup guide, and user guide waiting to empower your creative
endeavors!

Please [check out the documentation site](https://www.onnx-web.ai/docs/) for more info:

- [getting started guide](https://www.onnx-web.ai/docs/getting-started/)
- [setup guide](https://www.onnx-web.ai/docs/setup-guide/)
- [user guide](https://www.onnx-web.ai/docs/user-guide/)

![preview of txt2img tab using SDXL to generate ghostly astronauts eating weird hamburgers on an abandoned space station](./docs/readme-sdxl.png)

## Features

This is an incomplete list of new and interesting features:

- supports SDXL and SDXL Turbo
- wide variety of schedulers: DDIM, DEIS, DPM SDE, Euler Ancestral, LCM, UniPC, and more
- hardware acceleration on both AMD and Nvidia
  - tested on CUDA, DirectML, and ROCm
  - [half-precision support for low-memory GPUs](docs/user-guide.md#optimizing-models-for-lower-memory-usage) on both
    AMD and Nvidia
  - software fallback for CPU-only systems
- web app to generate and view images
  - [hosted on Github Pages](https://ssube.github.io/onnx-web), from your CDN, or locally
  - [persists your recent images and progress as you change tabs](docs/user-guide.md#image-history)
  - queue up multiple images and retry errors
  - translations available for English, French, German, and Spanish (please open an issue for more)
- supports many `diffusers` pipelines
  - [txt2img](docs/user-guide.md#txt2img-tab)
  - [img2img](docs/user-guide.md#img2img-tab)
  - [inpainting](docs/user-guide.md#inpaint-tab), with mask drawing and upload
  - [panorama](docs/user-guide.md#panorama-pipeline), for both SD v1.5 and SDXL
  - [upscaling](docs/user-guide.md#upscale-tab), with ONNX acceleration
- [add and use your own models](docs/user-guide.md#adding-your-own-models)
  - [convert models from diffusers and SD checkpoints](docs/converting-models.md)
  - [download models from HuggingFace hub, Civitai, and HTTPS sources](docs/user-guide.md#model-sources)
- blend in additional networks
  - [permanent and prompt-based blending](docs/user-guide.md#permanently-blending-additional-networks)
  - [supports LoRA and LyCORIS weights](docs/user-guide.md#lora-tokens)
  - [supports Textual Inversion concepts and embeddings](docs/user-guide.md#textual-inversion-tokens)
    - each layer of the embeddings can be controlled and used individually
- ControlNet
  - image filters for edge detection and other methods
  - with ONNX acceleration
- highres mode
  - runs img2img on the results of the other pipelines
  - multiple iterations can produce 8k images and larger
- [multi-stage](docs/user-guide.md#prompt-stages) and [region prompts](docs/user-guide.md#region-tokens)
  - seamlessly combine multiple prompts in the same image
  - provide prompts for different areas in the image and blend them together
  - change the prompt for highres mode and refine details without recursion
- infinite prompt length
  - [with long prompt weighting](docs/user-guide.md#long-prompt-weighting)
- [image blending mode](docs/user-guide.md#blend-tab)
  - combine images from history
- upscaling and correction
  - upscaling with Real ESRGAN, SwinIR, and Stable Diffusion
  - face correction with CodeFormer and GFPGAN
- [API server can be run remotely](docs/server-admin.md)
  - REST API can be served over HTTPS or HTTP
  - background processing for all image pipelines
  - polling for image status, plays nice with load balancers
- OCI containers provided
  - for all supported hardware accelerators
  - includes both the API and GUI bundle in a single container
  - runs well on [RunPod](https://www.runpod.io/), [Vast.ai](https://vast.ai/), and other GPU container hosting services

## Contents

- [onnx-web](#onnx-web)
  - [Features](#features)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Adding your own models](#adding-your-own-models)
  - [Usage](#usage)
    - [Known errors and solutions](#known-errors-and-solutions)
    - [Running the containers](#running-the-containers)
  - [Credits](#credits)

## Setup

There are a few ways to run onnx-web:

- cross-platform:
  - [clone this repository, create a virtual environment, and run `pip install`](docs/setup-guide.md#cross-platform-method)
  - [pulling and running the OCI containers](docs/server-admin.md#running-the-containers)
- on Windows:
  - [clone this repository and run one of the `setup-*.bat` scripts](docs/setup-guide.md#windows-python-installer)
  - [download and run the experimental all-in-one bundle](docs/setup-guide.md#windows-all-in-one-bundle)

You only need to run the server and should not need to compile anything. The client GUI is hosted on Github Pages and
is included with the Windows all-in-one bundle.

The extended setup docs have been [moved to the setup guide](docs/setup-guide.md).

### Adding your own models

You can [add your own models](./docs/user-guide.md#adding-your-own-models) by downloading them from the HuggingFace Hub
or Civitai or by converting them from local files, without making any code changes. You can also download and blend in
additional networks, such as LoRAs and Textual Inversions, using [tokens in the
prompt](docs/user-guide.md#prompt-tokens).

## Usage

### Known errors and solutions

Please see [the Known Errors section of the user guide](https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md#known-errors).

### Running the containers

This has [been moved to the server admin guide](docs/server-admin.md#running-the-containers).

## Credits

Some of the conversion and pipeline code was copied or derived from code in:

- [`Amblyopius/Stable-Diffusion-ONNX-FP16`](https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16)
  - GPL v3: https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16/blob/main/LICENSE
  - https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16/blob/main/pipeline_onnx_stable_diffusion_controlnet.py
  - https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16/blob/main/pipeline_onnx_stable_diffusion_instruct_pix2pix.py
- [`d8ahazard/sd_dreambooth_extension`](https://github.com/d8ahazard/sd_dreambooth_extension)
  - Non-commerical license: https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/license.md
  - https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/dreambooth/sd_to_diff.py
- [`huggingface/diffusers`](https://github.com/huggingface/diffusers)
  - Apache v2: https://github.com/huggingface/diffusers/blob/main/LICENSE
  - https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
- [`uchuusen/onnx_stable_diffusion_controlnet`](https://github.com/uchuusen/onnx_stable_diffusion_controlnet)
  - GPL v3: https://github.com/uchuusen/onnx_stable_diffusion_controlnet/blob/main/LICENSE
- [`uchuusen/pipeline_onnx_stable_diffusion_instruct_pix2pix`](https://github.com/uchuusen/pipeline_onnx_stable_diffusion_instruct_pix2pix)
  - Apache v2: https://github.com/uchuusen/pipeline_onnx_stable_diffusion_instruct_pix2pix/blob/main/LICENSE

Those parts have their own licenses with additional restrictions on commercial usage, modification, and redistribution.
The rest of the project is provided under the MIT license, and I am working to isolate these components into a library.

There are many other good options for using Stable Diffusion with hardware acceleration, including:

- https://github.com/Amblyopius/AMD-Stable-Diffusion-ONNX-FP16
- https://github.com/azuritecoin/OnnxDiffusersUI
- https://github.com/ForserX/StableDiffusionUI
- https://github.com/pingzing/stable-diffusion-playground
- https://github.com/quickwick/stable-diffusion-win-amd-ui

Getting this set up and running on AMD would not have been possible without guides by:

- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269
- https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs
- https://www.travelneil.com/stable-diffusion-updates.html
