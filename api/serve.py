from diffusers import OnnxStableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from flask import Flask, make_response, request, send_file
from stringcase import snakecase
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

# schedulers
scheduler_list = {
  'ddpm': DDPMScheduler.from_pretrained(model_path, subfolder="scheduler"),
  'ddim': DDIMScheduler.from_pretrained(model_path, subfolder="scheduler"),
  'pndm': PNDMScheduler.from_pretrained(model_path, subfolder="scheduler"),
  'lms-discrete': LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler"),
  'euler-a': EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler"),
  'euler': EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler"),
  'dpm-multi': DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler"),
}

def setup():
  if not path.exists(model_path):
    raise RuntimeError('model path must exist')
  if not path.exists(output_path):
    makedirs(output_path)

# setup
setup()
app = Flask(__name__)

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
  scheduler = scheduler_list[request.args.get('scheduler', 'euler-a')]

  print("txt2img from %s: %s/%s, %sx%s, %s" % (user, cfg, steps, width, height, prompt))
  image_queue.add(user)

  pipe = OnnxStableDiffusionPipeline.from_pretrained(
    model_path,
    provider="DmlExecutionProvider",
    safety_checker=None,
    scheduler=scheduler
  )
  image = pipe(
    prompt,
    height,
    width,
    num_inference_steps=steps,
    guidance_scale=cfg
  ).images[0]

  output = '%s/txt2img-%s' % (output_path, snakecase(prompt))
  print("txt2img output: %s" % (output))
  image.save(output)

  img_io = BytesIO()
  image.save(img_io, 'PNG', quality=100)
  img_io.seek(0)

  image_queue.remove(user)

  res = make_response(send_file(img_io, mimetype='image/png'))
  res.headers.add('Access-Control-Allow-Origin', '*')
  return res