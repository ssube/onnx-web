# Server Administration

This is the server administration guide for onnx-web.

Please see [the user guide](user-guide.md) for descriptions of the client and each of the parameters.

## Contents

- [Server Administration](#server-administration)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Running the containers](#running-the-containers)
    - [Configuring and running the server](#configuring-and-running-the-server)
      - [Securing the server](#securing-the-server)
    - [Updating the server](#updating-the-server)
    - [Building the client](#building-the-client)
    - [Hosting the client](#hosting-the-client)
    - [Customizing the client config](#customizing-the-client-config)
  - [Configuration](#configuration)
    - [Debug Mode](#debug-mode)
    - [Environment Variables](#environment-variables)
      - [Client Variables](#client-variables)
      - [Conversion Variables](#conversion-variables)
      - [Memory and Optimization Variables](#memory-and-optimization-variables)
      - [Path Variables](#path-variables)
      - [Platform Variables](#platform-variables)
      - [Server and Other Variables](#server-and-other-variables)
    - [Feature Flags](#feature-flags)
    - [Pipeline Optimizations](#pipeline-optimizations)
    - [Server Parameters](#server-parameters)
  - [Containers](#containers)
    - [CPU](#cpu)
    - [CUDA](#cuda)
    - [ROCm](#rocm)

## Setup

### Running the containers

OCI images are available for both the API and GUI, `ssube/onnx-web-api` and `ssube/onnx-web-gui`, respectively. These
are regularly built from the `main` branch and for all tags.

While two containers are provided, the API container also includes the GUI bundle. In most cases, you will only need to
run the API container. You may need both if you are hosting the API and GUI from separate pods or on different machines.

When using the containers, make sure to mount the `models/` and `outputs/` directories. The models directory can be
read-only, but outputs should be read-write.

```shell
> podman run -p 5000:5000 --rm -v ../models:/models:ro -v ../outputs:/outputs:rw docker.io/ssube/onnx-web-api:main-buster

> podman run -p 8000:80 --rm docker.io/ssube/onnx-web-gui:main-nginx-bullseye
```

The `ssube/onnx-web-gui` image is available in both Debian and Alpine-based versions, but the `ssube/onnx-web-api`
image is only available as a Debian-based image, due to [this Github issue with `onnxruntime`](https://github.com/microsoft/onnxruntime/issues/2909#issuecomment-593591317).

### Configuring and running the server

The server relies mostly on two paths, the models and outputs. It will make sure both paths exist when it starts up,
and will exit with an error if the models path does not.

Both of those paths exist in the git repository, with placeholder files to make sure they exist. You should not have to
create them, if you are using the default settings. You can customize the paths by setting `ONNX_WEB_MODEL_PATH` and
`ONNX_WEB_OUTPUT_PATH`, if your models exist somewhere else or you want output written to another disk, for example.

From within the `api/` directory, run the Flask server with the launch script:

```shell
# on Linux:
> ./launch.sh

# on Windows:
> launch.bat
```

This will allow access from other machines on your local network, but does not automatically make the server
accessible from the internet. You can access the server through the IP address printed in the console.

If you _do not_ want to allow access to the server from other machines on your local network, run the Flask server
_without_ the `--host` argument:

```shell
> flask --app=onnx_web.serve run
```

You can stop the server by pressing `Ctrl+C`.

#### Securing the server

When making the server publicly visible, make sure to use appropriately restrictive firewall rules along with it, and
consider using a web application firewall to help prevent malicious requests.

### Updating the server

Make sure to update your server occasionally. New features in the GUI may not be available on older servers, leading to
options being ignored or menus not loading correctly.

To update the server, make sure you are on the `main` branch and pull the latest version from Github:

```shell
> git branch
* main

> git pull
```

If you want to run a specific tag of the server, run `git checkout v0.12.0` with the desired tag.

### Building the client

If you plan on building the GUI bundle, instead of using a hosted version [like on Github Pages](https://ssube.github.io/onnx-web),
you will also need to install NodeJS 18:

- https://nodejs.org/en/download/

If you are using Windows and Git Bash, you may not have `make` installed. You can [add some of the missing tools](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058) from [the `ezwinports` project](https://sourceforge.net/projects/ezwinports/files/) and others.

From within the `gui/` directory, edit the `gui/examples/config.json` file so that `api.root` matches the URL printed
out by the `flask run` command you ran earlier. It should look something like this:

```json
{
  "api": {
    "root": "http://127.0.0.1:5000"
  }
}
```

Still in the `gui/` directory, build the UI bundle and run the dev server with Node:

```shell
> npm install -g yarn   # update the package manager

> make bundle

> node serve.js
```

### Hosting the client

You should be able to access the web interface at http://127.0.0.1:8000/index.html or your local machine's hostname.

- If you get a `Connection Refused` error, make sure you are using the correct address and the dev server is still running.
- If you get a `File not found` error, make sure you have built the UI bundle (`make bundle`) and are using the `/index.html` path

The txt2img tab will be active by default, with an example prompt. When you press the `Generate` button, an image should
appear on the page 10-15 seconds later (depending on your GPU and other hardware). Generating images on CPU will take
substantially longer, at least 2-3 minutes. The last four images will be shown, along with the parameters used to
generate them.

### Customizing the client config

You can customize the config file if you want to change the default model, platform (hardware acceleration), scheduler,
and prompt. If you have a good base prompt or always want to use the CPU fallback, you can set that in the config file:

```json
{
  "default": {
    "model": "stable-diffusion-onnx-v1-5",
    "platform": "amd",
    "scheduler": "euler-a",
    "prompt": "an astronaut eating a hamburger"
  }
}
```

When running the dev server, `node serve.js`, the config file will be loaded from `out/config.json`. If you want to load
a different config file, save it to your home directory named `onnx-web-config.json` and copy it into the output
directory after building the bundle:

```shell
> make bundle && cp -v ~/onnx-web-config.json out/config.json
```

When running the container, the config will be loaded from `/usr/share/nginx/html/config.json` and you can mount a
custom config using:

```shell
> podman run -p 8000:80 --rm -v ~/onnx-web-config.json:/usr/share/nginx/html/config.json:ro docker.io/ssube/onnx-web-gui:main-nginx-bullseye
```

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

#### Client Variables

- `ONNX_WEB_CIVITAI_ROOT`
  - root URL for downloading Civitai models
  - rarely needs to be changed
- `ONNX_WEB_CIVITAI_TOKEN`
  - Civitai API token for models that require login
  - you can create an API token in the Account settings page: https://civitai.com/user/account
- `ONNX_WEB_HUGGINGFACE_TOKEN`
  - Huggingface API token for models that require login

#### Conversion Variables

- `ONNX_WEB_CONVERT_CONTROL`
  - convert ControlNet
  - disable to skip ControlNet, saving some disk and memory
- `ONNX_WEB_CONVERT_EXTRACT`
  - extract models to Torch directory before converting to ONNX
  - disable to skip extraction, saving some disk
  - extraction uses an older code path that does not work with some newer models and has a non-commercial license
- `ONNX_WEB_CONVERT_RELOAD`
  - load ONNX pipelines after conversion to ensure they are valid
- `ONNX_WEB_CONVERT_SHARE_UNET`
  - unload UNet after converting and reload before ControlNet conversion
- `ONNX_WEB_CONVERT_OPSET`
  - ONNX opset used when converting models
- `ONNX_WEB_CONVERT_VALIDATE`
  - perform validation after model conversion
  - disable to accelerate conversion and save memory
- `ONNX_WEB_CONVERT_CPU_ONLY`
  - perform conversion on the CPU, even if a CUDA GPU is available
  - can allow conversion of models that do not fit in VRAM

#### Memory and Optimization Variables

- `ONNX_WEB_CACHE_MODELS`
  - the number of recent models to keep in memory
  - setting this to 0 will disable caching and free VRAM between images
- `ONNX_WEB_MEMORY_LIMIT`
  - memory limit for CUDA devices
  - does not apply to other platforms
- `ONNX_WEB_OPTIMIZATIONS`
  - comma-delimited list of optimizations to enable

#### Path Variables

- `ONNX_WEB_BUNDLE_PATH`
  - path where client bundle files can be found
- `ONNX_WEB_EXTRA_MODELS`
  - extra model files to be loaded
  - one or more filenames or paths, to JSON or YAML files matching [the extras schema](https://github.com/ssube/onnx-web/blob/main/api/schemas/extras.yaml)
- `ONNX_WEB_LOGGING_PATH`
  - path to `logging.yaml` config file
- `ONNX_WEB_MODEL_PATH`
  - path where models can be found
- `ONNX_WEB_OUTPUT_PATH`
  - path where output images should be saved
- `ONNX_WEB_PARAMS_PATH`
  - path to the directory where the `params.json` file can be found

#### Platform Variables

- `ONNX_WEB_ANY_PLATFORM`
  - whether or not to include the `any` option in the platform list
- `ONNX_WEB_BLOCK_PLATFORMS`
  - comma-delimited list of platforms that should not be presented to users
  - further filters the list of available platforms returned by ONNX runtime
  - can be used to prevent CPU generation on shared servers
- `ONNX_WEB_DEFAULT_PLATFORM`
  - the default platform to show in the client
  - overrides the `params.json` file

#### Server and Other Variables

- `ONNX_WEB_ADMIN_TOKEN`
  - token for admin operations
  - required to update extras file and restart workers
- `ONNX_WEB_CORS_ORIGIN`
  - comma-delimited list of allowed origins for CORS headers
- `ONNX_WEB_DEBUG`
  - wait for debugger to be attached before starting server
- `ONNX_WEB_EXTRA_ARGS`
  - extra arguments to the launch script
  - set this to `--half` to convert models to fp16
- `ONNX_WEB_FEATURE_FLAGS`
  - enable some [feature flags](#feature-flags)
- `ONNX_WEB_IMAGE_FORMAT`
  - output image file format
  - should be one of `jpeg` or `png`
- `ONNX_WEB_JOB_LIMIT`
  - number of jobs to run before restarting workers
  - can help prevent memory leaks
- `ONNX_WEB_PLUGINS`
  - comma-delimited list of plugin modules to load
- `ONNX_WEB_SERVER_VERSION`
  - server version
  - can be customized to identify nodes or pods in their logs
- `ONNX_WEB_SHOW_PROGRESS`
  - show progress bars in the logs
  - disabling this can reduce noise in server logs, especially when logging to a file
- `ONNX_WEB_WORKER_RETRIES`
  - number of times to retry a tile or stage before failing the image
  - more retries takes longer but can help prevent intermittent OOM errors from ruining long pipelines

### Feature Flags

- `panorama-highres`
  - when using the panorama pipeline with highres, prefer panorama views over stage tiling

### Pipeline Optimizations

- `diffusers-*`
  - `diffusers-attention-slicing`
    - https://huggingface.co/docs/diffusers/optimization/fp16#sliced-attention-for-additional-memory-savings
  - `diffusers-cpu-offload-*`
    - `diffusers-cpu-offload-sequential`
      - not available for ONNX pipelines (most of them)
      - https://huggingface.co/docs/diffusers/optimization/fp16#offloading-to-cpu-with-accelerate-for-memory-savings
    - `diffusers-cpu-offload-model`
      - not available for ONNX pipelines (most of them)
      - https://huggingface.co/docs/diffusers/optimization/fp16#model-offloading-for-fast-inference-and-memory-savings
  - `diffusers-memory-efficient-attention`
    - requires [the `xformers` library](https://huggingface.co/docs/diffusers/optimization/xformers)
    - https://huggingface.co/docs/diffusers/optimization/fp16#memory-efficient-attention
  - `diffusers-vae-slicing`
    - not available for ONNX pipelines (most of them)
    - https://huggingface.co/docs/diffusers/optimization/fp16#sliced-vae-decode-for-larger-batches
- `onnx-*`
  - `onnx-cpu-*`
    - CPU offloading for individual models
    - `onnx-cpu-text-encoder`
      - recommended for SDXL highres
    - `onnx-cpu-unet`
      - not recommended
    - `onnx-cpu-vae`
      - may be necessary for SDXL highres
  - `onnx-deterministic-compute`
    - enable ONNX deterministic compute
  - `onnx-fp16`
    - convert model nodes to 16-bit floating point values internally while leaving 32-bit inputs
  - `onnx-graph-*`
    - `onnx-graph-disable`
      - disable all ONNX graph optimizations
    - `onnx-graph-basic`
      - enable basic ONNX graph optimizations
    - `onnx-graph-all`
      - enable all ONNX graph optimizations
  - `onnx-low-memory`
    - disable ONNX features that allocate more memory than is strictly required or keep memory after use
- `torch-*`
  - `torch-fp16`
    - use 16-bit floating point values when converting and running pipelines
    - applies during conversion as well
    - only available on CUDA platform

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
