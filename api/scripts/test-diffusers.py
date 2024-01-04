from diffusers import OnnxStableDiffusionPipeline
from os import path

cfg = 8
steps = 22
height = 512
width = 512

model = path.join('..', 'models', 'stable-diffusion-onnx-v1-5')
prompt = 'an astronaut eating a hamburger'
output = path.join('..', 'outputs', 'test.png')

print('generating test image...')
pipe = OnnxStableDiffusionPipeline.from_pretrained(model, provider='DmlExecutionProvider', safety_checker=None)
image = pipe(prompt, height, width, num_inference_steps=steps, guidance_scale=cfg).images[0]
image.save(output)
print('saved test image to %s' % output)
