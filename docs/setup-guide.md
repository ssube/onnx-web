# Setup Guide

This guide covers the setup process for onnx-web, including downloading the Windows bundle.

## Contents

- [Setup Guide](#setup-guide)
  - [Contents](#contents)
  - [Windows all-in-one bundle](#windows-all-in-one-bundle)
  - [Windows Python installer](#windows-python-installer)
  - [Windows Store Python](#windows-store-python)

## Windows all-in-one bundle

1. Download the latest ZIP file from TODO
2. Find the ZIP file and `Extract All` to a memorable folder
3. Open the folder where you extracted the files
   1. You can add models to the `models` folder
   2. Your images will be in the `outputs` folder, along with files containing the parameters used to generate them
4. Run the local server using one of the `onnx-web-*.bat` scripts
   1. Run `onnx-web-half.bat` if you are using a GPU and you have < 12GB of VRAM
      - `-half` mode is compatible with both AMD and Nvidia GPUs
      - `-half` mode is not compatible with CPU mode
   2. Run `onnx-web-full.bat` if you are using CPU mode or if you have >= 16GB of VRAM
      - Try the `onnx-web-half.bat` script if you encounter out-of-memory errors or generating images is very slow
5. Wait for the models to be downloaded and converted
   1. Most models are distributed in PyTorch format and need to be converted into ONNX format
   2. This only happens once for each model and takes a few minutes
6. Open one of the URLs shown in the logs in your browser
   1. This will typically be http://127.0.0.1:5000?api=http://127.0.0.1:5000
   2. If you running the server on a different PC and not accessing it from a browser on the same system, use that PC's
      IP address instead of 127.0.0.1
   3. Any modern browser should work, including Chrome, Edge, and Firefox
   4. Mobile browsers also work, but have stricter mixed-content policies

## Windows Python installer

1. Install Git
2. Install Python 3.10
3. Open command prompt
4. Run one of the `setup-*.bat` scripts
   1. Run `setup-amd.bat` if you are using an AMD GPU and DirectML
   2. Run `setup-nvidia.bat` if you are using an Nvidia GPU and CUDA
   3. Run `setup-cpu.bat` if you are planning on only using CPU mode
5. In the future, run `launch.bat`
   1. You should only need to run the setup script once
   2. If you encounter any errors with Python imports, run the setup script again

## Windows Store Python

TODO
