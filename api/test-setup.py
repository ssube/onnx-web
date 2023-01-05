from diffusers import OnnxStableDiffusionPipeline

cfg = 8
steps = 22
height = 512
width = 512
prompt = "an astronaut eating a hamburger"

pipe = OnnxStableDiffusionPipeline.from_pretrained("../models/stable-diffusion-onnx-v1-5", provider="DmlExecutionProvider", safety_checker=None)
image = pipe(prompt, height, width, num_inference_steps=steps, guidance_scale=cfg).images[0]
image.save("../test.png")