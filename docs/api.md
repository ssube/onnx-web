# API

## Contents

- [API](#api)
  - [Contents](#contents)
  - [Endpoints](#endpoints)
    - [GUI bundle](#gui-bundle)
      - [`GET /`](#get-)
      - [`GET /<path>`](#get-path)
    - [Jobs](#jobs)
      - [`POST /api/job`](#post-apijob)
      - [`PUT /api/job/cancel`](#put-apijobcancel)
      - [`GET /api/job/status`](#get-apijobstatus)
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

### Jobs

#### `POST /api/job`

Create a new job that will run a chain pipeline provided in the `POST` body.

#### `PUT /api/job/cancel`

Cancel one or more jobs by name, provided in the `jobs` query parameter.

Request:

```shell
> curl http://localhost:5000/api/job/status?jobs=job-1,job-2,job-3,job-4
```

Response:

```json
[
  {
    "name": "job-1",
    "status": "cancelled"
  },
  {
    "name": "job-2",
    "status": "pending"
  }
]
```

#### `GET /api/job/status`

Get the status of one or more jobs by name, provided in the `jobs` query parameter.

Request:

```shell
> curl http://localhost:5000/api/job/status?jobs=job-1,job-2,job-3,job-4
```

Response:

```json
[
    {
        "metadata": [
            // metadata for each output
        ],
        "name": "job-1",
        "outputs": [
            "txt2img_job_1.png"
        ],
        "queue": {
            "current": 0,
            "total": 0
        },
        "stages": {
            "current": 4,
            "total": 6
        },
        "status": "running",
        "steps": {
            "current": 120,
            "total": 78
        },
        "tiles": {
            "current": 0,
            "total": 16
        }
    },
    {
        "name": "job-2",
        "queue": {
            "current": 2,
            "total": 3
        },
        "stages": {
            "current": 0,
            "total": 0
        },
        "status": "pending",
        "steps": {
            "current": 0,
            "total": 0
        },
        "tiles": {
            "current": 0,
            "total": 0
        }
    }
]
```

### Settings and parameters

#### `GET /api`

API introspection endpoint.

Returns a JSON document with all of the available API endpoints and valid methods for them:

```json
{
  "name": "onnx-web",
  "routes": [
    {
      "methods": [
        "HEAD",
        "GET",
        "OPTIONS"
      ],
      "path": "/static/:filename"
    },
    {
      "methods": [
        "HEAD",
        "GET",
        "OPTIONS"
      ],
      "path": "/"
    },
    ...
  ]
}
```

#### `GET /api/settings/filters`

List available mask and source filters.

#### `GET /api/settings/masks`

List available mask filters.

#### `GET /api/settings/models`

List available models, all types.

#### `GET /api/settings/noises`

List available noise sources.

#### `GET /api/settings/params`

Server parameters, slider min/max and version check.

#### `GET /api/settings/pipelines`

List available pipelines.

#### `GET /api/settings/platforms`

Available hardware acceleration platforms.

#### `GET /api/settings/schedulers`

Available pipeline schedulers.

#### `GET /api/settings/strings`

Server strings, from extras file and plugins.

#### `GET /api/settings/wildcards`

List available wildcard paths.

### Pipelines

#### `GET /api/ready`

Check if a pipeline has completed.

#### `POST /api/blend`

Blend two images using a mask.

#### `POST /api/chain`

Run a [custom chain pipeline](./chain-pipelines.md).

#### `POST /api/img2img`

Run an img2img pipeline.

#### `POST /api/inpaint`

Run an inpainting pipeline.

#### `POST /api/txt2img`

Run a txt2img pipeline.

#### `POST /api/txt2txt`

**Unstable.**

Run a txt2txt pipeline.

#### `POST /api/upscale`

Run an upscale pipeline.

#### `PUT /api/cancel`

Cancel a pending or in-progress job.

### Outputs

#### `GET /output/<path>`

Serve output images.

In debug mode, this will also include some intermediate images:

- `last-mask.png`
- `last-noise.png`
- `last-source.png`
- `last-tile-N.png`
