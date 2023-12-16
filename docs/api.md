# API

## Contents

- [API](#api)
  - [Contents](#contents)
  - [Endpoints](#endpoints)
    - [GUI bundle](#gui-bundle)
      - [`GET /`](#get-)
      - [`GET /<path>`](#get-path)
    - [Settings and parameters](#settings-and-parameters)
      - [`GET /api`](#get-api)
      - [`GET /api/settings/masks`](#get-apisettingsmasks)
      - [`GET /api/settings/models`](#get-apisettingsmodels)
      - [`GET /api/settings/noises`](#get-apisettingsnoises)
      - [`GET /api/settings/params`](#get-apisettingsparams)
      - [`GET /api/settings/platforms`](#get-apisettingsplatforms)
      - [`GET /api/settings/schedulers`](#get-apisettingsschedulers)
    - [Pipelines](#pipelines)
      - [`GET /api/ready`](#get-apiready)
      - [`POST /api/img2img`](#post-apiimg2img)
      - [`POST /api/inpaint`](#post-apiinpaint)
      - [`POST /api/outpaint`](#post-apioutpaint)
      - [`POST /api/txt2img`](#post-apitxt2img)
    - [Outputs](#outputs)
      - [`GET /output/<path>`](#get-outputpath)

## Endpoints

### GUI bundle

#### `GET /`

Serve `index.html`.

#### `GET /<path>`

Serve requested bundle file.

Usually includes:

- `bundle/main.js`
- `config.json`
- `index.html`

### Settings and parameters

#### `GET /api`

Introspection endpoint.

#### `GET /api/settings/masks`

Available mask filters.

#### `GET /api/settings/models`

Available models, all types.

#### `GET /api/settings/noises`

Available noise sources.

#### `GET /api/settings/params`

Server parameters, slider min/max and version check.

#### `GET /api/settings/platforms`

Available hardware acceleration platforms.

#### `GET /api/settings/schedulers`

Available pipeline schedulers.

### Pipelines

#### `GET /api/ready`

Check if a pipeline has completed.

#### `POST /api/img2img`

Run an img2img pipeline.

#### `POST /api/inpaint`

Run an inpainting pipeline.

#### `POST /api/outpaint`

Run an outpainting pipeline.

This uses the inpainting pipeline with more parameters and image filtering.

#### `POST /api/txt2img`

Run a txt2img pipeline.

### Outputs

#### `GET /output/<path>`

Serve output images.

In debug mode, this will also include some intermediate images:

- `last-mask.png`
- `last-noise.png`
- `last-source.png`
