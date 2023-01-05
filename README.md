# ONNX Web UI

This is a rudimentary web UI for ONNX models, providing a way to run GPU-accelerated models on Windows and AMD with a
remote web interface.

Based on work by:

- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269
- https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs
- https://www.travelneil.com/stable-diffusion-updates.html

## Contents

- [ONNX Web UI](#onnx-web-ui)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Install Git and Python](#install-git-and-python)
    - [Create a Virtual Environment](#create-a-virtual-environment)
    - [Install pip packages](#install-pip-packages)
    - [Install ORT Nightly package](#install-ort-nightly-package)
    - [Download and Convert Models](#download-and-convert-models)
  - [Usage](#usage)
    - [Configuring and running the server](#configuring-and-running-the-server)
      - [Securing the server](#securing-the-server)
    - [Configuring and hosting the client](#configuring-and-hosting-the-client)
    - [Using the web interface](#using-the-web-interface)

## Setup

This is a very similar process to what harishanand95 and averad's gists recommend, split up into a few larger steps:

1. Create a virtual environment
2. Install pip package

### Install Git and Python

Install Git and Python 3.10 for your environment:

- https://www.python.org/downloads/
- https://gitforwindows.org/

The latest version of git should be fine. Python must be 3.10 or earlier, 3.10 seems to work well.

### Create a Virtual Environment

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
> ./onnx_env/bin/activate

# on windows:
> .\onnx_env\Scripts\Activate.bat
```

### Install pip packages

Update pip itself:

```shell
> python -m pip install --upgrade pip
```

Install the following packages for AI:

```shell
> pip install diffusers transformers ftfy spacy scipy
> pip install onnx onnxruntime torch
> pip install onnxruntime-directml --force-reinstall    # TODO: is this one necessary?
```

Install the following packages for the web UI:

```shell
> pip install flask stringcase
```

### Install ORT Nightly package

Download the latest DirectML ORT nightly package for your version of Python and install it with pip.

Downloads can be found at https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly. If you are using
Python 3.10, download the `cp310` package. For Python 3.9, download the `cp39` package, and so on.

```shell
> wget https://aiinfra.pkgs.visualstudio.com/PublicPackages/_apis/packaging/feeds/7982ae20-ed19-4a35-a362-a96ac99897b7/pypi/packages/ort-nightly-directml/versions/1.14.dev20221214001/ort_nightly_directml-1.14.0.dev20221214001-cp310-cp310-win_amd64.whl/content
> pip install ~/Downloads/ort_nightly_directml-1.14.0.dev20221214001-cp310-cp310-win_amd64.whl --force-reinstall
```

Make sure to include the `--force-reinstall` flag, since it requires some older versions of other packages, and will
overwrite the versions you currently have installed.

### Download and Convert Models

Sign up for an account at https://huggingface.co and find the models you want to use. Popular options include:

- https://huggingface.co/runwayml/stable-diffusion-v1-5

Download the conversion script from the `huggingface/diffusers` repository:

- https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py

```shell
> wget https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
```

Run the conversion script with your desired model(s):

```shell
> python convert_stable_diffusion_checkpoint_to_onnx.py --model_path="runwayml/stable-diffusion-v1-5" --output_path="./stable-diffusion-onnx-v1-5"
```

This will take a little while to convert each model. Stable diffusion v1.4 is about 6GB, v1.5 is at least 10GB or so.

You can verify that all of the steps up to this point worked correctly by attempting to run the basic `txt2img` script
provided with `diffusers` and included here as `api/setup-test.py`.

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
