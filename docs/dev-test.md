# Development and Testing

## Contents

- [Development and Testing](#development-and-testing)
  - [Contents](#contents)
  - [API Development](#api-development)
    - [Debugging](#debugging)
    - [Models and Pipelines](#models-and-pipelines)
    - [Memory Profiling](#memory-profiling)
    - [Style](#style)
      - [Log Levels](#log-levels)
  - [GUI Development](#gui-development)
    - [Updating Github Pages](#updating-github-pages)
    - [Watch mode](#watch-mode)
  - [Testing](#testing)
    - [Pre-Release Test Plan](#pre-release-test-plan)
    - [Known Issues](#known-issues)

## API Development

Run `make ci` to run lint and the tests.

### Debugging

Launching the API with `debugpy` and `DEBUG=TRUE` may fail, because Flask creates a second process to watch for code
changes and they both try to bind the debug port: https://github.com/Microsoft/ptvsd/issues/1131

- https://code.visualstudio.com/docs/python/debugging#_debugging-by-attaching-over-a-network-connection
- https://github.com/microsoft/debugpy#debugging-a-script-file

### Models and Pipelines

Loading models and pipelines can be expensive. They should be converted and exported once, then cached per-process
whenever reasonably possible.

Most pipeline stages will have a corresponding load function somewhere, like `upscale_stable_diffusion` and
`load_stable_diffusion`. The load function should compare its parameters and reuse the existing pipeline when that is
possible without causing memory access errors. Most logging from the load function should be `debug` level.

### Memory Profiling

To track memory usage and leaks, run the API under `fil`:

```shell
> fil-profile run -m flask --app=onnx_web.serve run --host=0.0.0.0

> fil-profile run -m waitress --listen=0.0.0.0:5000 onnx_web.serve:app
```

Using `memray` will break the CUDA bridge or driver somehow, and prevents hardware acceleration from working. That
makes it extremely time consuming to test any kind of memory leak.

### Style

- all logs must use `logger` from top of file
  - every file should have a `logger = getLogger(__name__)` or equivalent before any real code

#### Log Levels

`onnx-web` uses the Python `logging` library and five base log levels, with an additional `TRACE` log level.

- `trace`
  - values needed to debug errors with a specific model
  - tensor shapes, list sizes, dict keys
  - only level that should log _expected_ values, especially empty values
  - may emit logs at idle, can produce MB/s under load
- `debug`
  - detailed debugging messages
- `info`
  - standard informative logs
- `warning`
  - error-ish messages that the code was able to handle without failing
- `error`
  - error messages that caused a request to fail, the app to crash, etc
- `exception`
  - not really a level, logs an error with a stacktrace, which can be helpful

When choosing log levels, consider what you will need to debug an issue from a remote server using the default
settings (`info`). The `trace`

## GUI Development

Run `make ci` to run lint, the tests, and build the bundle.

### Updating Github Pages

Checkout the `gh-pages` branch and run the `copy-bundle.sh` script, assuming you have the project
checked out to a directory named `onnx-web`.

You can also clone the GH pages branch into its own directory to avoid switching between them.

### Watch mode

Run `make watch` or `WATCH=TRUE node esbuild.js` to run esbuild in watch mode with a development server.

## Testing

### Pre-Release Test Plan

This is the test plan for manual pre-release testing and should exercise all of the major features.

Issues:

- TODO

Merges:

- TODO

Testing:

- txt2img
  - 256x256 with SD v1.5
    - [ ] should fail: neon blobs
    - has automation
  - 512x512 with SD v1.5
    - DEIS Multi
      - [ ] should work
    - DPM Multi
      - [ ] should work
      - has automation
    - Euler A
      - [ ] should work
    - Heun
      - [ ] should work
      - has automation
  - 512x512 with SD v2.1
    - [ ] should work
    - has automation
  - 768x768 with SD v2.1
    - [ ] should work, given sufficient memory
    - has automation
  - extra models
    - 512x512 with Knollingcase
      - [ ] should work
      - has automation
    - 512x512 with OpenJourney
      - [ ] should work
      - has automation
    - 256x256 with OpenJourney
      - [ ] should work
- img2img
  - 256x256 input
    - [ ] should fail: neon blobs
    - has automation
  - 512x512 input
    - [ ] should work
    - has automation
  - 1024x768 input
    - [ ] should work
- inpaint
  - regular inpaint
    - black mask
      - [ ] should keep all pixels, same image
      - has automation
    - white mask
      - [ ] should replace all pixels, different image
      - has automation
  - outpaint
    - 0 all sides
      - [ ] should work, run 1 tile
      - primarily a client-side test
    - 256 all sides
      - [ ] should work, run 8 tiles
      - has automation
    - 512 top and bottom, 0 left and right
      - [ ] should work, run 3 tiles
      - has automation
    - 512 left and right, 0 top and bottom
      - [ ] should work, run 3 tiles
      - has automation
- upscale
  - Real ESRGAN
    - x4 with CodeFormer
      - [ ] should work
    - x4 with GFPGAN
      - [ ] should work
    - x4 without face correction
      - [ ] should work
      - has automation
    - x2 without face correction
      - [ ] should work
      - has automation
    - x2 model and x4 scale
      - [ ] should sort of work: ignores scale and uses x2
    - x4 model and x2 scale
      - [ ] should fail: tiles
    - v3 model and x4 scale
      - [ ] should work
  - Stable Diffusion
    - using x2 scale
      - [ ] should fail: tiles
    - using x4 scale
      - [ ] should work
    - with CodeFormer
      - [ ] should work
    - with GFPGAN
      - [ ] should work
    - without face correction
      - [ ] should work
- blend
  - two 512x512 inputs
    - [ ] should work
  - two 1024x1024 inputs
    - [ ] should work
  - two different size inputs
    - 256x256 and 512x512
      - [ ] should work
    - 512x512 and 1024x1024
      - [ ] should work
- interactions
  - generate a new image
    - [ ] should request and then load an image from the server
  - delete a pending image
    - [ ] should remove a single image and leave the rest
  - delete a finished image
    - [ ] should remove a single image and leave the rest
  - copy an image to img2img
    - [ ] should switch to the img2img tab
    - [ ] should populate the image source
    - [ ] the generate button should be enabled
  - copy an image to inpaint
    - [ ] should switch to the inpaint tab
    - [ ] should populate the image source
    - [ ] the generate button should be enabled
  - copy an image to upscale
    - [ ] should switch to the upscale tab
    - [ ] should populate the image source
    - [ ] the generate button should be enabled
  - copy two images to blend
    - [ ] should switch to the blend tab
    - [ ] should populate the image sources
    - [ ] the generate button should be enabled once both sources have been populated
  - state should persist on refresh
    - [ ] loading images
    - [ ] switching tabs
    - [ ] images sources do not
- schedulers
  - [ ] DDIM
  - [ ] DDPM
  - [ ] DEIS Multi
  - [ ] DPM Multi
  - [ ] DPM Single
  - [ ] Euler A
  - [ ] Euler
  - [ ] Heun
  - [ ] iPNDM
  - [ ] KDPM2 A
  - [ ] KDPM2
  - [ ] Karras Ve
  - [ ] LMS
  - [ ] PNDM

Release:

- [ ] check and fix lint
- [ ] update package versions and stage files
- [ ] run `commit-and-tag-version --sign --git-tag-fallback --commit-all --release-as=minor` to make VERSION
- [ ] make sure packages and images have been built
- [ ] update GH pages bundle and default version
- [ ] post release on GH
- [ ] make follow up tickets
- [ ] close milestone and checklist

Repeat with and without LPW enabled. Feature combinations marked `should work` must produce a valid image for the
prompt, within a reasonable margin of creative freedom. Feature combinations marked `should fail` are known to produce
neon blobs, out of place tiles, and errors.

### Known Issues

- images of 256x256 or smaller will produce neon blobs
- inpaint does not work with LPW
