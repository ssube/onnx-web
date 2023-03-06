test_host="${1:-'http://127.0.0.1:5000'}"
test_images=0

while true;
do
  curl "${test_host}/api/txt2img?"\
'cfg=16.00&steps=30&scheduler=ddim&seed=-1&'\
'prompt=an+astronaut+eating+a+hamburger&negativePrompt=&'\
'model=stable-diffusion-onnx-v1-5&platform=any&'\
'upscaling=upscaling-real-esrgan-x2-plus&correction=correction-codeformer&'\
'lpw=false&width=512&height=512&upscaleOrder=correction-both' \
    -X 'POST' \
    --compressed \
    --insecure || break;
  ((test_images++));
  echo "waiting after $test_images";
  sleep 3;
done
