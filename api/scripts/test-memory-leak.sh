test_host="${1:-127.0.0.1}"
test_images=0

while true;
do
  curl "http://${test_host}:5000/api/txt2img?"\
'cfg=16.00&steps=3&scheduler=deis-multi&seed=-1&'\
'prompt=an+astronaut+eating+a+hamburger&negativePrompt=&'\
'model=stable-diffusion-onnx-v1-5&platform=any&'\
'upscaling=upscaling-real-esrgan-x2-plus&correction=correction-codeformer&'\
'lpw=false&width=512&height=512&upscaleOrder=correction-both' \
    -X 'POST' \
    --compressed \
    --insecure || break;
  ((test_images++));
  echo "waiting after $test_images";
  sleep 30;
done
