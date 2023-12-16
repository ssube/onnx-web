# Getting Started With onnx-web

onnx-web is a tool for generating images with Stable Diffusion pipelines, including SDXL.

## Contents

- [Getting Started With onnx-web](#getting-started-with-onnx-web)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Windows bundle setup](#windows-bundle-setup)
    - [Other setup methods](#other-setup-methods)
  - [Running](#running)
    - [Running the server](#running-the-server)
    - [Running the web UI](#running-the-web-ui)
  - [Tabs](#tabs)
    - [Txt2img Tab](#txt2img-tab)
    - [Img2img Tab](#img2img-tab)
    - [Inpaint Tab](#inpaint-tab)
    - [Upscale Tab](#upscale-tab)
    - [Blend Tab](#blend-tab)
    - [Models Tab](#models-tab)
    - [Settings Tab](#settings-tab)
  - [Image parameters](#image-parameters)
    - [Common image parameters](#common-image-parameters)
    - [Unique image parameters](#unique-image-parameters)
  - [Prompt syntax](#prompt-syntax)
    - [LoRAs and embeddings](#loras-and-embeddings)
    - [CLIP skip](#clip-skip)
  - [Highres](#highres)
    - [Highres prompt](#highres-prompt)
    - [Highres iterations](#highres-iterations)
  - [Profiles](#profiles)
    - [Loading from files](#loading-from-files)
    - [Saving profiles in the web UI](#saving-profiles-in-the-web-ui)
    - [Sharing parameters profiles](#sharing-parameters-profiles)
  - [Panorama pipeline](#panorama-pipeline)
    - [Region prompts](#region-prompts)
    - [Region seeds](#region-seeds)
  - [Grid mode](#grid-mode)
    - [Grid tokens](#grid-tokens)
  - [Memory optimizations](#memory-optimizations)
    - [Converting to fp16](#converting-to-fp16)
    - [Moving models to the CPU](#moving-models-to-the-cpu)

## Setup

### Windows bundle setup

1. Download
2. Extract
3. Security flags
4. Run

### Other setup methods

Link to the other methods.

## Running

### Running the server

Run server or bundle.

### Running the web UI

Open web UI.

Use it from your phone.

## Tabs

There are 5 tabs, which do different things.

### Txt2img Tab

Words go in, pictures come out.

### Img2img Tab

Pictures go in, better pictures come out.

ControlNet lives here.

### Inpaint Tab

Pictures go in, parts of the same picture come out.

### Upscale Tab

Just highres and super resolution.

### Blend Tab

Use the mask tool to combine two images.

### Models Tab

Add and manage models.

### Settings Tab

Manage web UI settings.

Reset buttons.

## Image parameters

### Common image parameters

- Scheduler
- Eta
  - for DDIM
- CFG
- Steps
- Seed
- Batch size
- Prompt
- Negative prompt
- Width, height

### Unique image parameters

- UNet tile size
- UNet overlap
- Tiled VAE
- VAE tile size
- VAE overlap

See the complete user guide for details about the highres, upscale, and correction parameters.

## Prompt syntax

### LoRAs and embeddings

`<lora:filename:1.0>` and `<inversion:filename:1.0>`.

### CLIP skip

`<clip:skip:2>` for anime.

## Highres

### Highres prompt

`txt2img prompt || img2img prompt`

### Highres iterations

Highres will apply the upscaler and highres prompt (img2img pipeline) for each iteration.

The final size will be `scale ** iterations`.

## Profiles

Saved sets of parameters for later use.

### Loading from files

- load from images
- load from JSON

### Saving profiles in the web UI

Use the save button.

### Sharing parameters profiles

Use the download button.

Share profiles in the Discord channel.

## Panorama pipeline

### Region prompts

`<region:X:Y:W:H:S:F_TLBR:prompt+>`

### Region seeds

`<reseed:X:Y:W:H:?:F_TLBR:seed>`

## Grid mode

Makes many images. Takes many time.

### Grid tokens

`__column__` and `__row__` if you pick token in the menu.

## Memory optimizations

### Converting to fp16

Enable the option.

### Moving models to the CPU

Option for each model.
