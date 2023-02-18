# Development and Testing

## Contents

- [Development and Testing](#development-and-testing)
  - [Contents](#contents)
  - [API Development](#api-development)
    - [Style](#style)
    - [Models and Pipelines](#models-and-pipelines)
  - [GUI Development](#gui-development)
    - [Updating Github Pages](#updating-github-pages)
    - [Watch mode](#watch-mode)
  - [Testing](#testing)
    - [Pre-Release Test Plan](#pre-release-test-plan)
    - [Known Issues](#known-issues)

## API Development

Run `make ci` to run lint and the tests.

### Style

- all logs must use `logger` from top of file
  - every file should have a `logger = getLogger(__name__)` or equivalent before any real code

### Models and Pipelines

Loading models and pipelines can be expensive. They should be converted and exported once, then cached per-process
whenever reasonably possible.

Most pipeline stages will have a corresponding load function somewhere, like `upscale_stable_diffusion` and `load_stable_diffusion`. The load function should compare its parameters and reuse the existing pipeline when
that is possible without causing memory access errors. Most logging from the load function should be `debug` level.

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
  - 512x512 with SD v1.5
    - DEIS Multi
      - [ ] should work
    - DPM Multi
      - [ ] should work
    - Euler A
      - [ ] should work
  - 512x512 with SD v2.1
    - [ ] should work
  - 768x768 with SD v2.1
    - [ ] should work, given sufficient memory
  - extra models
    - 512x512 with Knollingcase
      - [ ] should work
    - 512x512 with OpenJourney
      - [ ] should work
    - 256x256 with OpenJourney
      - [ ] should work
- img2img
  - 256x256 input
    - [ ] should fail: neon blobs
  - 512x512 input
    - [ ] should work
  - 1024x768 input
    - [ ] should work
- inpaint
  - regular inpaint
    - black mask
      - [ ] should keep all pixels, same image
    - white mask
      - [ ] should replace all pixels, different image
  - outpaint
    - 0 all sides
      - [ ] should work, run 1 tile
    - 256 all sides
      - [ ] should work, run 8 tiles
    - 512 top and bottom, 0 left and right
      - [ ] should work, run 3 tiles
    - 512 left and right, 0 top and bottom
      - [ ] should work, run 3 tiles
- upscale
  - Real ESRGAN
    - x4 with CodeFormer
      - [ ] should work
    - x4 with GFPGAN
      - [ ] should work
    - x4 without face correction
      - [ ] should work
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
