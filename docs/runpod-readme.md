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

To automatically download and convert models from [HuggingFace](https://huggingface.co), issue a read-only token from the [Access Tokens page](https://huggingface.co/settings/tokens) and set it as the value of the `HF_TOKEN` variable in this pod.
