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
  - 512x512 with SD v1.5
    - DPM Multi
    - Euler A
  - 768x768 with SD v2.1
- img2img
  - 256x256 input
  - 512x512 input
  - 1024x768 input
- inpaint
  - outpaint
    - 256 all sides
    - 512 top and bottom, 0 left and right
    - 512 left and right, 0 top and bottom
- upscale
  - Real ESRGAN
    - with face correction
    - without face correction
  - Stable Diffusion
    - with face correction
    - without face correction
- interactions
  - generate new image
  - delete pending image
  - delete finished image
  - copy to img2img
  - copy to inpaint
