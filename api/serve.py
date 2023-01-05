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
from stringcase import spinalcase
from io import BytesIO
from os import environ, path, makedirs

# defaults
default_prompt = "a photo of an astronaut eating a hamburger"
default_height = 512
default_width = 512
default_steps = 20
default_cfg = 8

max_height = 512
max_width = 512
max_steps = 150
max_cfg = 30

# paths
model_path = environ.get('ONNX_WEB_MODEL_PATH', "../../stable_diffusion_onnx")
output_path = environ.get('ONNX_WEB_OUTPUT_PATH', "../../web_output")

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

def get_and_clamp(args, key, default_value, max_value, min_value=1):
  return min(max(int(args.get(key, default_value)), min_value), max_value)

def get_from_map(args, key, values, default):
  selected = args.get(key, default)
  if selected in values:
    return values[selected]
  else:
    return values[default]

# setup
if not path.exists(model_path):
  raise RuntimeError('model path must exist')

if not path.exists(output_path):
  makedirs(output_path)

app = Flask(__name__)

# routes
@app.route('/')
def hello():
    return 'Hello, %s' % (__name__)

@app.route('/txt2img')
def txt2img():
  user = request.remote_addr

  cfg = get_and_clamp(request.args, 'cfg', default_cfg, max_cfg)
  height = get_and_clamp(request.args, 'height', default_height, max_height)
  prompt = request.args.get('prompt', default_prompt)
  steps = get_and_clamp(request.args, 'steps', default_steps, max_steps)
  scheduler = get_from_map(request.args, 'scheduler', scheduler_list, 'euler-a')
  width = get_and_clamp(request.args, 'width', default_width, max_width)

  print("txt2img from %s: %s/%s, %sx%s, %s" % (user, cfg, steps, width, height, prompt))

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

  output = '%s/txt2img_%s.png' % (output_path, spinalcase(prompt[0:64]))
  print("txt2img output: %s" % (output))
  image.save(output)

  img_io = BytesIO()
  image.save(img_io, 'PNG', quality=100)
  img_io.seek(0)

  res = make_response(send_file(img_io, mimetype='image/png'))
  res.headers.add('Access-Control-Allow-Origin', '*')
  return res