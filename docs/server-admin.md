# Server Administration

This is the server administration guide for ONNX web.

Please see [the user guide](user-guide.md) for descriptions of the client and each of the parameters.

## Contents

- [Server Administration](#server-administration)
  - [Contents](#contents)
  - [Configuration](#configuration)
    - [Server Parameters](#server-parameters)
  - [Containers](#containers)
    - [CPU](#cpu)
    - [CUDA](#cuda)
    - [ROCm](#rocm)

## Configuration

Configuration is still very simple, loading models from a directory and parameters from a single JSON file.

### Server Parameters

You can limit the parameters in user requests to values within a reasonable range using the `params.json` file.

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
