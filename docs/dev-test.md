# Development and Testing

## Contents

- [Development and Testing](#development-and-testing)
  - [Contents](#contents)
  - [Development](#development)
    - [API](#api)
    - [GUI](#gui)
      - [Updating Github Pages](#updating-github-pages)
  - [Testing](#testing)
    - [Pre-Release Test Plan](#pre-release-test-plan)
    - [Known Issues](#known-issues)

## Development

### API

- TODO: testing
- TODO: lint/style

### GUI

Run `make ci` to build the bundle.

#### Updating Github Pages

Checkout the `gh-pages` branch and run the `copy-bundle.sh` script, assuming you have the project
checked out to a directory named `onnx-web`.

You can also clone the GH pages branch into its own directory to avoid switching between them.

## Testing

### Pre-Release Test Plan

This is the test plan for manual pre-release testing and should exercise all of the major features.

- txt2img
  - 256x256 with SD v1.5
    - should fail: neon blobs
  - 512x512 with SD v1.5
    - DPM Multi
      - should work
    - Euler A
      - should work
  - 512x512 with SD v1.5
  - 768x768 with SD v2.1
    - should work, given sufficient memory
  - extra models
    - Knollingcase
      - should work
    - OpenJourney
      - should work
- img2img
  - 256x256 input
    - TODO
  - 512x512 input
    - should work
  - 1024x768 input
    - should work
- inpaint
  - outpaint
    - 0 all sides
      - should work
    - 256 all sides
      - should work
    - 512 top and bottom, 0 left and right
      - should work
    - 512 left and right, 0 top and bottom
      - should work
- upscale
  - Real ESRGAN
    - with CodeFormer
      - should work
    - with GFPGAN
      - should work
    - without face correction
      - should work
    - using x2 model and x4 scale
      - should fail: tiles
    - using x4 model and x2 scale
      - should fail: tiles
    - using v3 model and x2 scale
    - using v3 model and x4 scale
  - Stable Diffusion
    - using x2 scale
      - should fail: tiles
    - using x4 scale
      - should work
    - with CodeFormer
      - should work
    - with GFPGAN
      - should work but doesn't: https://github.com/ssube/onnx-web/issues/87
    - without face correction
      - should work
- interactions
  - generate a new image
    - should request and then load an image from the server
  - delete a pending image
    - should remove a single image and leave the rest
  - delete a finished image
    - should remove a single image and leave the rest
  - copy an image to img2img
    - should switch to the img2img tab
    - should populate the image source
    - the generate button should be enabled
  - copy an image to inpaint
    - should switch to the inpaint tab
    - should populate the image source
    - the generate button should be enabled

Repeat with and without LPW enabled. Feature combinations marked `should work` must produce a valid image for the
prompt, within a reasonable margin of creative freedom. Feature combinations marked `should fail` are known to produce
neon blobs, out of place tiles, and errors.

### Known Issues

- images of 256x256 or smaller will produce neon blobs
- inpaint does not work with LPW
