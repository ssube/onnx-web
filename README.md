# ONNX Web UI

This is a rudimentary web UI for ONNX models, providing a way to run GPU-accelerated models on Windows and even AMD
with a remote web interface.

This is still fairly early and instructions are a little rough, but it works on my machine. If I keep working on this
for more than a week, I would like to add img2img and Nvidia support.

![txt2img with example astronaut prompt and image](./docs/readme-preview.png)

Based on work by:

- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269
- https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs
- https://www.travelneil.com/stable-diffusion-updates.html

## Features

- REST API server capable of running ONNX models with DirectML acceleration
  - multiple schedulers
- web app to generate and view images
  - parameter inputs with validation
  - txt2img mode

## Contents

- [ONNX Web UI](#onnx-web-ui)
  - [Features](#features)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Note about setup paths](#note-about-setup-paths)
    - [Install Git and Python](#install-git-and-python)
    - [Create a virtual environment](#create-a-virtual-environment)
    - [Install pip packages](#install-pip-packages)
    - [Install ORT Nightly package](#install-ort-nightly-package)
    - [Download and Convert Models](#download-and-convert-models)
  - [Usage](#usage)
    - [Configuring and running the server](#configuring-and-running-the-server)
      - [Securing the server](#securing-the-server)
    - [Configuring and hosting the client](#configuring-and-hosting-the-client)
    - [Using the web interface](#using-the-web-interface)

## Setup

This is a very similar process to what [harishanand95](https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269)
and [averad's](https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674) gists recommend, split up into a few
steps:

1. Install Git and Python, if you have not already
2. Create a virtual environment
3. Install pip packages
4. Install ORT Nightly package
5. Download and convert models

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

### Install Git and Python

Install Git and Python 3.10 for your environment:

- https://www.python.org/downloads/
- https://gitforwindows.org/

The latest version of git should be fine. Python must be 3.10 or earlier, 3.10 seems to work well.

### Create a virtual environment

Make sure you have Python 3.10 or earlier, then create a virtual environment:

```shell
> python --version
Python 3.10

> pip install virtualenv

> python -m venv onnx_env
```

This will contain all of the pip libraries. If you update or reinstall Python, you will need to recreate the virtual
environment.

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

Install the following packages for AI:

```shell
> pip install numpy>=1.20,<1.24   # version is important, 1.24 removed the deprecated np.float symbol
> pip install accelerate diffusers transformers ftfy spacy scipy
> pip install onnx onnxruntime torch
```

If you are running on Windows, install the DirectML ONNX runtime as well:

```shell
> pip install onnxruntime-directml --force-reinstall
```

Install the following packages for the web UI:

```shell
> pip install flask stringcase
```

_Or_ install all of these packages at once using [the `requirements.txt` file](./api/requirements.txt):

```shell
> pip install -r requirements.txt
```

At the moment, only `numpy` seems to need a specific version. If you see an error about `np.float`, make sure you are
not using `numpy>=1.24`. [This SO question](https://stackoverflow.com/questions/74844262/how-to-solve-error-numpy-has-no-attribute-float-in-python)
has more details.

I got a warning about an incompatibility in `protobuf` when installing the `onnxruntime-directml` package, but have not seen any issues. Some of the gist guides recommend `diffusers=0.3.0`, but I had trouble with old versions of `diffusers`
before 0.6.0 or so. If I can determine a good set of working versions, I will pin them in `requirements.txt`.

### Install ORT Nightly package

Download the latest DirectML ORT nightly package for your version of Python and install it with pip.

Downloads can be found at https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly. If you are using
Python 3.10, download the `cp310` package. For Python 3.9, download the `cp39` package, and so on.

```shell
> pip install ~/Downloads/ort_nightly_directml-1.14.0.dev20221214001-cp310-cp310-win_amd64.whl --force-reinstall
```

Make sure to include the `--force-reinstall` flag, since it requires some older versions of other packages, and will
overwrite the versions you currently have installed.

### Download and Convert Models

Sign up for an account at https://huggingface.co and find the models you want to use. Popular options include:

- https://huggingface.co/runwayml/stable-diffusion-v1-5

Log into the HuggingFace CLI:

```shell
> huggingface-cli.exe login
```

Issue an API token from https://huggingface.co/settings/tokens, naming it something memorable like `onnx-web`, and then
paste it into the prompt.

Download the conversion script from the `huggingface/diffusers` repository to the root of this project:

- https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py

Run the conversion script with your desired model(s):

```shell
> python convert_stable_diffusion_checkpoint_to_onnx.py \
    --model_path="runwayml/stable-diffusion-v1-5" \
    --output_path="./models/stable-diffusion-onnx-v1-5"
```

This will take a little while to convert each model. Stable diffusion v1.4 is about 6GB, v1.5 is at least 10GB or so.

You can verify that all of the steps up to this point worked correctly by attempting to run the `api/setup-test.py`
script, which is a slight variation on the original txt2img script.

## Usage

### Configuring and running the server

In the `api/` directory, run the server with Flask:

```shell
> flask --app serve run
```

Note the IP address this prints.

If you want to access the server from other machines on your local network, pass the `--host` argument:

```shell
> flask --app serve run --host 0.0.0.0
```

This will listen for requests from your current local network and may be dangerous.

#### Securing the server

When making the server publicly visible, make sure to use appropriately restrictive firewall rules along with it, and
consider using a web application firewall to help prevent malicious requests.

### Configuring and hosting the client

From within the `gui/` directory, edit the `gui/examples/config.json` file so that `api.root` is the URL printed out by
the `flask run` command from earlier. It should look something like this:

```json
{
  "api": {
    "root": "http://127.0.0.1:5000"
  }
}
```

Still in the `gui/` directory, run the dev server with Node:

```shell
> node serve.js
```

### Using the web interface

You should be able to access the web interface at http://127.0.0.1:3000/index.html or your local machine's hostname.

The txt2img tab will be active by default, with an example prompt. You can press the `Generate` button and an image
should appear on the page 10-15 seconds later (depending on your GPU and other hardware).
