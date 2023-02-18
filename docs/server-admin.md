# Server Administration

This is the server administration guide for ONNX web.

Please see [the user guide](user-guide.md) for descriptions of the client and each of the parameters.

## Contents

- [Server Administration](#server-administration)
  - [Contents](#contents)
  - [Configuration](#configuration)
    - [Debug Mode](#debug-mode)
    - [Environment Variables](#environment-variables)
    - [Server Parameters](#server-parameters)
  - [Containers](#containers)
    - [CPU](#cpu)
    - [CUDA](#cuda)
    - [ROCm](#rocm)

## Configuration

Configuration is still very simple, loading models from a directory and parameters from a single JSON file. Some
additional configuration can be done through environment variables starting with `ONNX_WEB`.

### Debug Mode

Setting the `DEBUG` variable to any value except `false` will enable debug mode, which will print garbage
collection details and save some extra images to disk.

The images are:

- `output/last-mask.png`
  - the last `mask` image submitted with an inpaint request
- `output/last-noise.png`
  - the last noise source generated for an inpaint request
- `output/last-source.png`
  - the last `source` image submitted with an img2img, inpaint, or upscale request

These extra images can be helpful when debugging inpainting, especially poorly blended edges or visible noise.

### Environment Variables

Paths:

- `ONNX_WEB_BUNDLE_PATH`
  - path where client bundle files can be found
- `ONNX_WEB_MODEL_PATH`
  - path where models can be found
- `ONNX_WEB_OUTPUT_PATH`
  - path where output images should be saved
- `ONNX_WEB_PARAMS_PATH`
  - path to the directory where the `params.json` file can be found

Others:

- `ONNX_WEB_ANY_PLATFORM`
  - whether or not to include the `any` option in the platform list
- `ONNX_WEB_BLOCK_PLATFORMS`
  - comma-delimited list of platforms that should not be presented to users
  - further filters the list of available platforms returned by ONNX runtime
  - can be used to prevent CPU generation on shared servers
- `ONNX_WEB_CACHE_MODELS`
  - the number of recent models to keep in memory
  - setting this to 0 will disable caching and free VRAM between images
- `ONNX_WEB_CORS_ORIGIN`
  - comma-delimited list of allowed origins for CORS headers
- `ONNX_WEB_DEFAULT_PLATFORM`
  - the default platform to show in the client
  - overrides the `params.json` file
- `ONNX_WEB_NUM_WORKERS`
  - number of background workers for image pipelines
  - this should be equal to or less than the number of available GPUs
- `ONNX_WEB_SHOW_PROGRESS`
  - show progress bars in the logs
  - disabling this can reduce noise in server logs, especially when logging to a file

### Server Parameters

You can limit the image parameters in user requests to a reasonable range using values in the `params.json` file.

The keys share the same name as the query string parameter, and the format for each numeric value is:

```json
{
  "default": 50,
  "min": 1,
  "max": 100,
  "step": 1
}
```

Setting the `step` to a decimal value between 0 and 1 will allow decimal inputs, but the client is hard-coded to send 2
decimal places in the query and only some parameters are parsed as floats, so values below `0.01` will effect the GUI
but not the output images, and some controls effectively force a step of `1`.

## Containers

### CPU

This is the simplest container to run and does not require any drivers or devices, but is also the slowest to
generate images.

### CUDA

Requires CUDA container runtime and 11.x driver on the host.

### ROCm

Requires ROCm driver on the host.

Run with podman using:

```shell
> podman run -it \
    --device=/dev/dri \
    --device=/dev/kfd \
    --group-add video \
    --security-opt seccomp=unconfined \
    -e ONNX_WEB_MODEL_PATH=/data/models \
    -e ONNX_WEB_OUTPUT_PATH=/data/outputs \
    -v /var/lib/onnx-web/models:/data/models:rw \
    -v /var/lib/onnx-web/outputs:/data/outputs:rw \
    -p 5000:5000 \
    docker.io/ssube/onnx-web-api:main-rocm-ubuntu
```

Rootless podman does not appear to work and will show a `root does not belong to group 'video'` error, which does
not make much sense on its own, but appears to refers to the user who launched the container.
