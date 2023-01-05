from diffusers import OnnxStableDiffusionPipeline
from flask import Flask

max_height = 512
max_width = 512

app = Flask(__name__)
pipe = OnnxStableDiffusionPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider", safety_checker=None)


@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/txt2img')
def txt2img():
  height = request.args.get('height', max_height)
  width = request.args.get('width', max_width)
  prompt = request.args.get('prompt', "a photo of an astronaut eating a hamburger")
  steps = 50
  cfg = 8

  image = pipe(prompt, height, width, num_inference_steps=steps, guidance_scale=cfg).images[0]
  image.save("astronaut_rides_horse.png")
