from diffusers import OnnxStableDiffusionPipeline
from os import path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import time

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


upscale = path.join('..', 'outputs', 'test-large.png')
esrgan = path.join('..', 'models', 'RealESRGAN_x4plus.onnx')

print('upscaling test image...')
sess = ort.InferenceSession(esrgan, providers=['DmlExecutionProvider'])

in_image = cv2.imread(output, cv2.IMREAD_UNCHANGED)

in_mat = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
in_mat = np.transpose(in_mat, (2, 1, 0))[np.newaxis]
in_mat = in_mat.astype(np.float32)
in_mat = in_mat/255

start_time = time.time()
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
in_mat = torch.tensor(in_mat)
out_mat = sess.run([output_name], {input_name: in_mat.cpu().numpy()})[0]
elapsed_time = time.time() - start_time
print(elapsed_time)
print('upscaled test image to %s')