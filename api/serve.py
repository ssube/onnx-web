from diffusers import OnnxStableDiffusionPipeline
from flask import Flask, make_response, request, send_file
from io import BytesIO
from os import environ, path, makedirs

# defaults
empty_prompt = "a photo of an astronaut eating a hamburger"
max_height = 512
max_width = 512
max_steps = 50
max_cfg = 8

# paths
model_path = environ.get('ONNX_WEB_MODEL_PATH', "../../stable_diffusion_onnx")
output_path = environ.get('ONNX_WEB_OUTPUT_PATH', "../../web_output")

# queue
image_queue = set()

def setup():
  if not path.exists(model_path):
    raise RuntimeError('model path must exist')
  if not path.exists(output_path):
    makedirs(output_path)

# setup
setup()
app = Flask(__name__)
pipe = OnnxStableDiffusionPipeline.from_pretrained(model_path, provider="DmlExecutionProvider", safety_checker=None)

# routes
@app.route('/')
def hello():
    return 'Hello, %s' % (__name__)

@app.route('/txt2img')
def txt2img():
  if len(image_queue) > 0:
    return 'Queue full: %s' % (image_queue)

  user = request.remote_addr
  prompt = request.args.get('prompt', empty_prompt)
  height = request.args.get('height', max_height)
  width = request.args.get('width', max_width)
  steps = int(request.args.get('steps', max_steps))
  cfg = int(request.args.get('cfg', max_cfg))

  print("txt2img from %s: %s/%s, %sx%s, %s" % (user, cfg, steps, width, height, prompt))
  image_queue.add(user)

  image = pipe(prompt, height, width, num_inference_steps=steps, guidance_scale=cfg).images[0]
  # image.save("astronaut_rides_horse.png")

  img_io = BytesIO()
  image.save(img_io, 'PNG', quality=100)
  img_io.seek(0)

  image_queue.remove(user)

  res = make_response(send_file(img_io, mimetype='image/png'))
  res.headers.add('Access-Control-Allow-Origin', '*')
  return res