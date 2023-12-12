# ONNX Web

onnx-web is a tool for running Stable Diffusion and other [ONNX models](https://onnx.ai/) with hardware acceleration,
on both AMD and Nvidia GPUs and with a CPU software fallback.

The GUI is [hosted on Github Pages](https://ssube.github.io/onnx-web/) and runs in all major browsers, including on
mobile devices. It allows you to select the model and accelerator being used for each image pipeline. Image parameters
are shown for each of the major modes, and you can either upload or paint the mask for inpainting and outpainting. The
last few output images are shown below the image controls, making it easy to refer back to previous parameters or save
an image from earlier.

The API runs on both Linux and Windows and provides a REST API to run many of the pipelines from [`diffusers`
](https://huggingface.co/docs/diffusers/main/en/index), along with metadata about the available models and accelerators,
and the output of previous runs. Hardware acceleration is supported on both AMD and Nvidia for both Linux and Windows,
with a CPU fallback capable of running on laptop-class machines.

Please check out [the setup guide to get started](docs/setup-guide.md) and [the user guide for more
details](https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md).

![preview of txt2img tab using SDXL to generate ghostly astronauts eating weird hamburgers on an abandoned space station](./docs/readme-sdxl.png)

## Features

This is an incomplete list of new and interesting features, with links to the user guide:

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

- [ONNX Web](#onnx-web)
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
- [`uchuusen/pipeline_onnx_stable_diffusion_instruct_pix2pix](https://github.com/uchuusen/pipeline_onnx_stable_diffusion_instruct_pix2pix)
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
