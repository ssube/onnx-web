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
      - [`GET /api/settings/filters`](#get-apisettingsfilters)
      - [`GET /api/settings/masks`](#get-apisettingsmasks)
      - [`GET /api/settings/models`](#get-apisettingsmodels)
      - [`GET /api/settings/noises`](#get-apisettingsnoises)
      - [`GET /api/settings/params`](#get-apisettingsparams)
      - [`GET /api/settings/pipelines`](#get-apisettingspipelines)
      - [`GET /api/settings/platforms`](#get-apisettingsplatforms)
      - [`GET /api/settings/schedulers`](#get-apisettingsschedulers)
      - [`GET /api/settings/strings`](#get-apisettingsstrings)
      - [`GET /api/settings/wildcards`](#get-apisettingswildcards)
    - [Pipelines](#pipelines)
      - [`GET /api/ready`](#get-apiready)
      - [`POST /api/blend`](#post-apiblend)
      - [`POST /api/chain`](#post-apichain)
      - [`POST /api/img2img`](#post-apiimg2img)
      - [`POST /api/inpaint`](#post-apiinpaint)
      - [`POST /api/txt2img`](#post-apitxt2img)
      - [`POST /api/txt2txt`](#post-apitxt2txt)
      - [`POST /api/upscale`](#post-apiupscale)
      - [`PUT /api/cancel`](#put-apicancel)
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

#### `GET /api/settings/filters`

TODO

#### `GET /api/settings/masks`

Available mask filters.

#### `GET /api/settings/models`

Available models, all types.

#### `GET /api/settings/noises`

Available noise sources.

#### `GET /api/settings/params`

Server parameters, slider min/max and version check.

#### `GET /api/settings/pipelines`

TODO

#### `GET /api/settings/platforms`

Available hardware acceleration platforms.

#### `GET /api/settings/schedulers`

Available pipeline schedulers.

#### `GET /api/settings/strings`

TODO

#### `GET /api/settings/wildcards`

TODO

### Pipelines

#### `GET /api/ready`

Check if a pipeline has completed.

#### `POST /api/blend`

TODO

#### `POST /api/chain`

TODO

#### `POST /api/img2img`

Run an img2img pipeline.

#### `POST /api/inpaint`

Run an inpainting pipeline.

#### `POST /api/txt2img`

Run a txt2img pipeline.

#### `POST /api/txt2txt`

TODO

#### `POST /api/upscale`

TODO

#### `PUT /api/cancel`

TODO

### Outputs

#### `GET /output/<path>`

Serve output images.

In debug mode, this will also include some intermediate images:

- `last-mask.png`
- `last-noise.png`
- `last-source.png`
- `last-tile-N.png`
