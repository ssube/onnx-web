# ONNX Web

This is a web UI for running [ONNX models](https://onnx.ai/) with hardware acceleration on both AMD and Nvidia system,
with a CPU software fallback.

The API runs on both Linux and Windows and provides access to the major functionality of [`diffusers`](https://huggingface.co/docs/diffusers/main/en/index),
along with metadata about the available models and accelerators, and the output of previous runs. Hardware acceleration
is supported on both AMD and Nvidia for both Linux and Windows, with a CPU fallback capable of running on laptop-class
machines.

The GUI is [hosted on Github Pages](https://ssube.github.io/onnx-web/) and runs in all major browsers, including on
mobile devices. It allows you to select the model and accelerator being used for each image pipeline. Image parameters
are shown for each of the major modes, and you can either upload or paint the mask for inpainting and outpainting. The
last few output images are shown below the image controls, making it easy to refer back to previous parameters or save
an image from earlier.

Please [see the User Guide](https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md) for more details.

![txt2img with detailed knollingcase renders of a soldier in a cloudy alien jungle](./docs/readme-preview.png)

## Features

This is an incomplete list of new and interesting features, with links to the user guide:

- hardware acceleration on both AMD and Nvidia
  - [tested on CUDA, DirectML, and ROCm](#install-pip-packages)
  - [half-precision support for low-memory GPUs](docs/user-guide.md#optimizing-models-for-lower-memory-usage) on both
    AMD and Nvidia
  - software fallback for CPU-only systems
- web app to generate and view images
  - [hosted on Github Pages](https://ssube.github.io/onnx-web), from your CDN, or locally
  - [persists your recent images and progress as you change tabs](docs/user-guide.md#image-history)
  - queue up multiple images and retry errors
  - translations available for English, French, German, and Spanish (please open an issue for more)
- supports many `diffusers` pipelines
  - [txt2img](docs/user-guide.md#txt2img-tab)
  - [img2img](docs/user-guide.md#img2img-tab)
  - [inpainting](docs/user-guide.md#inpaint-tab), with mask drawing and upload
  - [upscaling](docs/user-guide.md#upscale-tab), with ONNX acceleration
- [add and use your own models](docs/user-guide.md#adding-your-own-models)
  - [convert models from diffusers and SD checkpoints](docs/converting-models.md)
  - [download models from HuggingFace hub, Civitai, and HTTPS sources](docs/user-guide.md#model-sources)
- blend in additional networks
  - [permanent and prompt-based blending](docs/user-guide.md#permanently-blending-additional-networks)
  - [supports LoRA weights](docs/user-guide.md#lora-tokens)
  - [supports Textual Inversion concepts and embeddings](docs/user-guide.md#textual-inversion-tokens)
- infinite prompt length
  - [with long prompt weighting](docs/user-guide.md#long-prompt-weighting)
  - expand and control Textual Inversions per-layer
- [image blending mode](docs/user-guide.md#blend-tab)
  - combine images from history
- upscaling and face correction
  - upscaling with Real ESRGAN or Stable Diffusion
  - face correction with CodeFormer or GFPGAN
- [API server can be run remotely](docs/server-admin.md)
  - REST API can be served over HTTPS or HTTP
  - background processing for all image pipelines
  - polling for image status, plays nice with load balancers
- OCI containers provided
  - for all supported hardware accelerators
  - includes both the API and GUI bundle in a single container
  - runs well on [RunPod](https://www.runpod.io/) and other GPU container hosting services

## Contents

- [ONNX Web](#onnx-web)
  - [Features](#features)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Install Git and Python](#install-git-and-python)
    - [Note about setup paths](#note-about-setup-paths)
    - [Create a virtual environment](#create-a-virtual-environment)
    - [Install pip packages](#install-pip-packages)
      - [For AMD on Linux: PyTorch ROCm and ONNX runtime ROCm](#for-amd-on-linux-pytorch-rocm-and-onnx-runtime-rocm)
      - [For AMD on Windows: PyTorch CPU and ONNX runtime DirectML](#for-amd-on-windows-pytorch-cpu-and-onnx-runtime-directml)
      - [For CPU everywhere: PyTorch CPU and ONNX runtime CPU](#for-cpu-everywhere-pytorch-cpu-and-onnx-runtime-cpu)
      - [For Nvidia everywhere: Install PyTorch GPU and ONNX GPU](#for-nvidia-everywhere-install-pytorch-gpu-and-onnx-gpu)
    - [Download and convert models](#download-and-convert-models)
      - [Converting your own models](#converting-your-own-models)
    - [Test the models](#test-the-models)
  - [Usage](#usage)
    - [Running the containers](#running-the-containers)
    - [Configuring and running the server](#configuring-and-running-the-server)
      - [Securing the server](#securing-the-server)
    - [Updating the server](#updating-the-server)
    - [Building the client](#building-the-client)
    - [Hosting the client](#hosting-the-client)
    - [Customizing the client config](#customizing-the-client-config)
    - [Known errors and solutions](#known-errors-and-solutions)
  - [Credits](#credits)

## Setup

To run the server and generate images, you need to [install Git and Python](#install-git-and-python) along with [a few
pip libraries](#install-pip-packages), then [run the conversion script](#converting-your-own-models) to download and
convert [the models you want to use](#download-and-convert-models).

### Install Git and Python

Install Git and Python 3.10 for your environment:

- https://gitforwindows.org/
- https://www.python.org/downloads/

The latest version of git should be fine. Python should be 3.9 or 3.10, although 3.8 and 3.11 may work if the correct
packages are available for your platform. If you already have Python installed for another form of Stable Diffusion,
that should work, but make sure to verify the version in the next step.

Make sure you have Python 3.9 or 3.10:

```shell
> python --version
Python 3.10
```

If your system differentiates between Python 2 and 3 and uses the `python3` and `pip3` commands for the Python 3.x
tools, make sure to adjust the commands shown here. They should otherwise be the same: `python3 --version`.

Once you have those basic packages installed, clone this git repository:

```shell
> git clone https://github.com/ssube/onnx-web.git
```

### Note about setup paths

This project contains both Javascript and Python, for the client and server respectively. Make sure you are in the
correct directory when working with each part.

Most of these setup commands should be run in the Python environment and the `api/` directory:

```shell
> cd api
> pwd
/home/ssube/code/github/ssube/onnx-web/api
```

The Python virtual environment will be created within the `api/` directory.

The Javascript client can be built and run within the `gui/` directory.

### Create a virtual environment

Change into the `api/` directory, then create a virtual environment:

```shell
> pip install virtualenv
> python -m venv onnx_env
```

This will contain all of the pip libraries. If you update or reinstall Python, you will need to recreate the virtual
environment.

If you receive an error like `Error: name 'cmd' is not defined`, there may be [a bug in the `venv` module](https://www.mail-archive.com/debian-bugs-dist@lists.debian.org/msg1884072.html) on certain
Debian-based systems. You may need to install venv through apt instead:

```shell
> sudo apt install python3-venv   # only if you get an error
```

Every time you start using ONNX web, activate the virtual environment:

```shell
# on linux:
> source ./onnx_env/bin/activate

# on windows:
> .\onnx_env\Scripts\Activate.bat
```

Update pip itself:

```shell
> python -m pip install --upgrade pip
```

### Install pip packages

You can install all of the necessary packages at once using [the `requirements/base.txt` file](./api/requirements/base.txt)
and the `requirements/` file for your platform. Install them in separate commands and make sure to install the
platform-specific packages first:

```shell
> pip install -r requirements/amd-linux.txt
> pip install -r requirements/base.txt
# or
> pip install -r requirements/amd-windows.txt
> pip install -r requirements/base.txt
# or
> pip install -r requirements/cpu.txt
> pip install -r requirements/base.txt
# or
> pip install -r requirements/nvidia.txt
> pip install -r requirements/base.txt
```

Only install one of the platform-specific requirements files, otherwise you may end up with the wrong version of
PyTorch or the ONNX runtime. The full list of available ONNX runtime packages [can be found here
](https://download.onnxruntime.ai/).

If you have successfully installed both of the requirements files for your platform, you do not need to install
any of the packages shown in the following platform-specific sections.

The ONNX runtime nightly packages used by the `requirements/*-nightly.txt` files can be substantially faster than the
last release, but may not always be stable. Many of the nightly packages are specific to one version of Python and
some are only available for Python 3.8 and 3.9, so you may need to find the correct package for your environment. If
you are using Python 3.10, download the `cp310` package. For Python 3.9, download the `cp39` package, and so on.
Installing with pip will figure out the correct package for you.

#### For AMD on Linux: PyTorch ROCm and ONNX runtime ROCm

If you are running on Linux with an AMD GPU, install the ROCm versions of PyTorch and `onnxruntime`:

```shell
> pip install "torch==1.13.1" "torchvision==0.14.1" --extra-index-url https://download.pytorch.org/whl/rocm5.2
# and one of
> pip install https://download.onnxruntime.ai/onnxruntime_training-1.14.1%2Brocm54-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# or
> pip install https://download.onnxruntime.ai/onnxruntime_training-1.15.0.dev20230326001%2Brocm542-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Make sure you have installed ROCm 5.x ([see their documentation
](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.2.3/page/How_to_Install_ROCm.html#_How_to_Install) for more
details) and that the version of `onnxruntime` matches your ROCm drivers. The version of PyTorch does not need to match
exactly, and they only have limited versions available.

Ubuntu 20.04 supports ROCm 5.2 and Ubuntu 22.04 supports ROCm 5.4, unless you want to build custom packages. The ROCm
5.x series supports many discrete AMD cards since the Vega 20 architecture, with [a partial list of supported cards
shown here](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3/page/Prerequisites.html#d5434e465).

#### For AMD on Windows: PyTorch CPU and ONNX runtime DirectML

If you are running on Windows with an AMD GPU, install the DirectML ONNX runtime as well:

```shell
> pip install "torch==1.13.1" "torchvision==0.14.1" --extra-index-url https://download.pytorch.org/whl/cpu
# and one of
> pip install onnxruntime-directml
# or
> pip install ort-nightly-directml --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --force-reinstall
```

If you DirectML package upgrades numpy to an incompatible version >= 1.24, downgrade it:

```shell
> pip install "numpy>=1.20,<1.24" --force-reinstall  # the DirectML package will upgrade numpy to 1.24, which will not work
```

You can optionally install the latest DirectML ORT nightly package, which may provide a substantial performance
increase.

#### For CPU everywhere: PyTorch CPU and ONNX runtime CPU

If you are running with a CPU and no hardware acceleration, install `onnxruntime` and the CPU version of PyTorch:

```shell
> pip install "torch==1.13.1" "torchvision==0.14.1" --extra-index-url https://download.pytorch.org/whl/cpu
# and
> pip install onnxruntime
# or
> pip install ort-nightly --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --force-reinstall
```

#### For Nvidia everywhere: Install PyTorch GPU and ONNX GPU

If you are running with an Nvidia GPU on any operating system, install `onnxruntime-gpu` and the CUDA version of
PyTorch:

```shell
> pip install "torch==1.13.1" "torchvision==0.14.1" --extra-index-url https://download.pytorch.org/whl/cu117
# and
> pip install onnxruntime-gpu
# or
> pip install ort-nightly-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --force-reinstall
```

Make sure you have installed CUDA 11.x and that the version of PyTorch matches the version of CUDA
([see their documentation](https://pytorch.org/get-started/locally/) for more details).

### Download and convert models

Sign up for an account at https://huggingface.co and find the models you want to use. Some popular options are
already listed in the `convert.py` script, including:

- https://huggingface.co/runwayml/stable-diffusion-v1-5
- https://huggingface.co/runwayml/stable-diffusion-inpainting
- https://huggingface.co/stabilityai/stable-diffusion-2-1
- https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
- https://huggingface.co/Aybeeceedee/knollingcase
- https://huggingface.co/prompthero/openjourney

You will need at least one of the base models for txt2img and img2img mode. If you want to use inpainting, you will
also need one of the inpainting models. The upscaling and face correction models are downloaded from Github by the
same script.

Log into the HuggingFace CLI:

```shell
# on linux:
> huggingface-cli login

# on Windows:
> huggingface-cli.exe login
```

Issue an API token from https://huggingface.co/settings/tokens, naming it something memorable like `onnx-web`, and then
paste it into the prompt.

Running the launch script from the `api/` directory will convert the base models before starting the API server:

```shell
# on Linux:
> ./launch.sh

# on Windows:
> launch.bat
```

Models that have already been downloaded and converted will be skipped, so it should be safe to run this script after
every update. Some additional, more specialized models are available using the `--extras` flag.

The conversion script has a few other options, which can be printed using `python -m onnx_web.convert --help`. If you
are using CUDA on Nvidia hardware, using the `--half` option may make things faster.

This will take a little while to convert each model. Stable diffusion v1.4 is about 6GB, v1.5 is at least 10GB or so.
You can skip certain models by including a `--skip names` argument if you want to save time or disk space. For example,
using `--skip stable-diffusion-onnx-v2-inpainting stable-diffusion-onnx-v2-1` will not download the Stable
Diffusion v2 models.

#### Converting your own models

You can include your own models in the conversion script without making any code changes and download additional
networks to blend during conversion or by using prompt tokens. For more details, please [see the user guide
](./docs/user-guide.md#adding-your-own-models).

Make a copy of the `api/extras.json` file and edit it to include the models you want to download and convert:

```json
{
  "diffusion": [
    {
      "name": "diffusion-knollingcase",
      "label": "Knollingcase",
      "source": "Aybeeceedee/knollingcase"
    },
    {
      "name": "diffusion-openjourney",
      "source": "prompthero/openjourney"
    },
    {
      "name": "diffusion-stablydiffused-aesthetic-v2-6",
      "label": "Stably Diffused Aesthetic Mix v2.6",
      "source": "civitai://6266?type=Pruned%20Model&format=SafeTensor",
      "format": "safetensors"
    },
    {
      "name": "diffusion-unstable-ink-dream-v6",
      "label": "Unstable Ink Dream v6",
      "source": "civitai://5796",
      "format": "safetensors"
    },
    {
      "name": "sonic-diffusion-v1-5",
      "source": "runwayml/stable-diffusion-v1-5",
      "inversions": [
        {
          "name": "ugly-sonic",
          "source": "huggingface://sd-concepts-library/ugly-sonic",
          "model": "concept"
        }
      ]
    }
  ],
  "correction": [],
  "upscaling": [],
  "networks": [
    {
      "name": "cubex",
      "source": "huggingface://sd-concepts-library/cubex",
      "label": "Cubex",
      "model": "concept",
      "type": "inversion"
    },
    {
      "name": "birb",
      "source": "huggingface://sd-concepts-library/birb-style",
      "label": "Birb",
      "model": "concept",
      "type": "inversion"
    },
    {
      "name": "minecraft",
      "source": "huggingface://sd-concepts-library/minecraft-concept-art",
      "label": "Minecraft Concept",
      "model": "concept",
      "type": "inversion"
    },
  ],
  "sources": []
}
```

Set the `ONNX_WEB_EXTRA_MODELS` environment variable to the path to your new `extras.json` file before running the
launch script:

```shell
# on Linux:
> export ONNX_WEB_EXTRA_MODELS="/home/ssube/onnx-web-extras.json"
> ./launch-extras.sh

# on Windows:
> set ONNX_WEB_EXTRA_MODELS=C:\Users\ssube\onnx-web-extras.json
> launch-extras.bat
```

Make sure to use the `launch-extras.sh` or `.bat` script if you want to convert the extra models, especially if you
have added your own.

### Test the models

You should verify that all of the steps up to this point have worked correctly by attempting to run the
`api/scripts/test-diffusers.py` script, which is a slight variation on the original txt2img script.

If the script works, there will be an image of an astronaut in `outputs/test.png`.

If you get any errors, check [the known errors section](#known-errors-and-solutions).

## Usage

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

If you want to run a specific tag of the server, run `git checkout v0.9.0` with the desired tag.

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

### Known errors and solutions

Please see [the Known Errors section of the user guide](https://github.com/ssube/onnx-web/blob/main/docs/user-guide.md#known-errors).

## Credits

Some of the conversion code was copied or derived from code in:

- https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
  - https://github.com/huggingface/diffusers/blob/main/LICENSE
- https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/dreambooth/sd_to_diff.py
  - https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/license.md

Those parts have their own license with additional restrictions and may need permission for commercial usage.

Getting this set up and running on AMD would not have been possible without guides by:

- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269
- https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs
- https://www.travelneil.com/stable-diffusion-updates.html

There are many other good options for using Stable Diffusion with hardware acceleration, including:

- https://github.com/Amblyopius/AMD-Stable-Diffusion-ONNX-FP16
- https://github.com/azuritecoin/OnnxDiffusersUI
- https://github.com/ForserX/StableDiffusionUI
- https://github.com/pingzing/stable-diffusion-playground
- https://github.com/quickwick/stable-diffusion-win-amd-ui
