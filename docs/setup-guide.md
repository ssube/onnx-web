# Setup Guide

This guide covers the setup process for onnx-web, including downloading the Windows bundle.

## Contents

- [Setup Guide](#setup-guide)
  - [Contents](#contents)
  - [Cross-platform method](#cross-platform-method)
    - [Install Git and Python](#install-git-and-python)
    - [Note about setup paths](#note-about-setup-paths)
    - [Create a virtual environment](#create-a-virtual-environment)
    - [Install pip packages](#install-pip-packages)
      - [For AMD on Linux: PyTorch ROCm and ONNX runtime ROCm](#for-amd-on-linux-pytorch-rocm-and-onnx-runtime-rocm)
      - [For AMD on Windows: PyTorch CPU and ONNX runtime DirectML](#for-amd-on-windows-pytorch-cpu-and-onnx-runtime-directml)
      - [For CPU everywhere: PyTorch CPU and ONNX runtime CPU](#for-cpu-everywhere-pytorch-cpu-and-onnx-runtime-cpu)
      - [For Nvidia everywhere: Install PyTorch GPU and ONNX GPU](#for-nvidia-everywhere-install-pytorch-gpu-and-onnx-gpu)
    - [Download and convert models](#download-and-convert-models)
    - [Test the models](#test-the-models)
    - [Download the web UI bundle](#download-the-web-ui-bundle)
    - [Launch the server](#launch-the-server)
    - [Open the web UI](#open-the-web-ui)
  - [Windows-specific methods](#windows-specific-methods)
    - [Windows all-in-one bundle](#windows-all-in-one-bundle)
    - [Windows Python installer](#windows-python-installer)

## Cross-platform method

This works on both Linux and Windows, for both AMD and Nvidia, but requires some familiarity with the command line.

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

Every time you start using onnx-web, activate the virtual environment:

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

You can install all of the necessary packages at once using [the `requirements/base.txt` file](https://github.com/ssube/onnx-web/blob/main/api/requirements/base.txt)
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
any of the packages shown in the following sections and you should [skip directly to testing the models](#test-the-models).

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

Before continuing, you will need to download or convert at least one Stable Diffusion model into the ONNX format.

Some pre-converted models are available in [the `models/preconverted-*.json`
files](https://github.com/ssube/onnx-web/blob/main/models/preconverted-base-fp32.json), including Stable Diffusion v1.5,
SDXL, and SDXL Turbo.

You can also download and extract the models yourself. For example:

```shell
> wget -O ../models/.cache/stable-diffusion-v1-5-fp32.zip https://models.onnx-files.com/stable-diffusion-v1-5-fp32.zip

...
Saving to: ‘../models/.cache/stable-diffusion-v1-5-fp32.zip’
...

> unzip ../models/.cache/stable-diffusion-v1-5-fp32.zip -d ../models/stable-diffusion-onnx-v1-5

Archive:  ../models/.cache/stable-diffusion-v1-5-fp32.zip
   creating: ../models/stable-diffusion-onnx-v1-5/vae_encoder/
  inflating: ../models/stable-diffusion-onnx-v1-5/vae_encoder/model.onnx
  inflating: ../models/stable-diffusion-onnx-v1-5/LICENSE.txt
  inflating: ../models/stable-diffusion-onnx-v1-5/README.txt
  ...

> file ../models/stable-diffusion-onnx-v1-5/model_config.json

../models/stable-diffusion-onnx-v1-5/model_index.json: JSON data
```

Note that the included `--base` models and the pre-converted models have different folder names. This is intentional,
so they don't conflict with each other during testing.

### Test the models

You should verify that all of the steps up to this point have worked correctly by attempting to run the
`api/scripts/test-diffusers.py` script, which is a slight variation on the original txt2img script.

If the script works, there will be an image of an astronaut in `outputs/test.png`.

If you get any errors, check [the known errors section of the user guide](user-guide.md#known-errors).

### Download the web UI bundle

Once the server environment is working, you will need the latest files for the web UI. This is a Javascript bundle and
you can download a pre-built copy from Github or compile your own.

From [the `gh-pages` branch](https://github.com/ssube/onnx-web/tree/gh-pages), select the version matching your server
and download all three files:

- `bundle/main.js`
- `config.json`
- `index.html`

Copy them into your local [`api/gui` folder](https://github.com/ssube/onnx-web/tree/main/api/gui). Make sure to keep the
`main.js` bundle in the `bundle` subfolder and copy the files into the `gui` folder within the `api` folder, _not_ the
`gui` folder in the root of the repository.

For example, for a v0.12.0 server, copy the files from https://github.com/ssube/onnx-web/tree/gh-pages/v0.12.0 into your
local copy of https://github.com/ssube/onnx-web/tree/main/api/gui and
https://github.com/ssube/onnx-web/tree/main/api/gui/bundle.

### Launch the server

After you have confirmed that you have a working environment, launch the server using one of the provided launch scripts
in the `api` folder:

```shell
# on Linux
> ./launch.sh

# on Windows
> .\launch.ps1
```

This will download and/or convert any missing models, then launch an API server on port `:5000`. If the server starts
up correctly, you should see an admin token in the logs:

```none
[2023-12-31 13:46:16,451] INFO: MainProcess MainThread onnx_web.main: all plugins loaded successfully
[2023-12-31 13:46:16,466] INFO: MainProcess MainThread onnx_web.server.load: available acceleration platforms: any - CPUExecutionProvider (None), cpu - CPUExecutionProvider (None)
[2023-12-31 13:46:16,494] INFO: MainProcess MainThread onnx_web.main: starting v0.12.0 API server with admin token: RANDOM-TOKEN
```

If you see any errors about the port already being in use, make sure you are not running any other servers or programs
that use port 5000, or change the `--port` argument in the launch script. If you change the port, make sure to use that
new port in any other commands that you run.

### Open the web UI

With the server running, open the web UI in your favorite web browser. If you are running the server locally, the UI will
be available at http://localhost:5000/.

If you are running the server on a different computer, you will need to use that computer's IP address or local DNS name
and provide that same address in the `?api` argument. For example, with a server running on a remote computer at
`10.2.2.100` and using port 5001, the URL would be `http://10.2.2.100:5001?api=http://10.2.2.100:5001`.

You can change the `?api` argument to use multiple servers while keeping your state and results. Note that changing the
server while an image is generating will cause it to fail in the web UI, since the new server will not be aware of that
image.

## Windows-specific methods

These methods are specific to Windows, tested on Windows 10, and still experimental. They should provide an easier
setup experience.

### Windows all-in-one bundle

1. Install the latest Visual C++ 2019 redistributable
   1. https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
   2. https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Download the latest ZIP file from [the apextoaster Nexus server](https://artifacts.apextoaster.com/#browse/browse:onnx-web-dist)
3. Find the ZIP file and `Extract All` to a memorable folder
4. Open the folder where you extracted the files
   1. Your models will be converted into the `models` folder, and you can [add your own models](user-guide.md#adding-your-own-models)
   2. Your images will be in the `outputs` folder, along with files containing the parameters used to generate them
5. Make sure the server is allowed to run
   1. Open the `server` folder
   2. Right-click the `onnx-web.exe` file and click `Properties`
   3. On the `General` tab, click `Unblock` next to the message `This file came from another computer and might be
      blocked to help protect this computer.`
   4. Go back to the folder where you extracted the files
   5. Repeat step 3 for the `onnx-web-*.bat` files
6. Run the local server using one of the `onnx-web-*.bat` scripts
   1. Run `onnx-web-half.bat` if you are using a GPU and you have < 12GB of VRAM
      - `-half` mode is compatible with both AMD and Nvidia GPUs
      - `-half` mode is not compatible with CPU mode
   2. Run `onnx-web-full.bat` if you are using CPU mode or if you have >= 16GB of VRAM
      - Try the `onnx-web-half.bat` script if you encounter out-of-memory errors or generating images is very slow
7. Wait for the models to be downloaded and converted
   1. Most models are distributed in PyTorch format and need to be converted into ONNX format
   2. This only happens once for each model and takes a few minutes
8. Open one of the URLs shown in the logs in your browser
   1. This will typically be http://127.0.0.1:5000?api=http://127.0.0.1:5000
   2. If you running the server on a different PC and not accessing it from a browser on the same system, use that PC's
      IP address instead of 127.0.0.1
   3. Any modern browser should work, including Chrome, Edge, and Firefox
   4. Mobile browsers also work, but have stricter mixed-content policies

### Windows Python installer

1. Install the latest Visual C++ 2019 redistributable
   1. https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
   2. https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install Git
   - https://gitforwindows.org/
3. Install Python 3.10
   - https://www.python.org/downloads/
4. Clone or download the onnx-web repository
   - `git clone https://github.com/ssube/onnx-web.git`
   - https://github.com/ssube/onnx-web/archive/refs/heads/main.zip
5. Open a command prompt window
6. Run one of the `setup-*.bat` scripts
   1. Run `setup-amd.bat` if you are using an AMD GPU and DirectML
   2. Run `setup-nvidia.bat` if you are using an Nvidia GPU and CUDA
   3. Run `setup-cpu.bat` if you are planning on only using CPU mode
7. After the first run, you can run `launch.bat` instead of the setup script
   1. You should only need to run the setup script once
   2. If you encounter any errors with Python imports, run the setup script again
