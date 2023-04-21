# onnx-web Windows bundle

This is the Windows all-in-one bundle for onnx-web, a tool for running Stable Diffusion and
other models using ONNX hardware acceleration: https://github.com/ssube/onnx-web

Please check the setup guide for the latest instructions, this may be an older version:
https://github.com/ssube/onnx-web/blob/main/docs/setup-guide.md#windows-all-in-one-bundle

## Running onnx-web

You can run the local server using one of the setup scripts, onnx-web-full.bat or onnx-web-half.bat. Use the
onnx-web-half.bat script if you are using a GPU and have < 12GB of VRAM. Use the onnx-web-full.bat script if
you are using CPU mode or if you have >= 16GB of VRAM.

The user interface should be available in your browser at http://127.0.0.1:5000?api=http://127.0.0.1:5000. If
your PC uses a different IP address or you are running the server on one PC and using it from another, use that IP
address instead.
