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

esrgan = path.join('..', 'models', 'RealESRGAN_x4plus.onnx')
output = path.join('..', 'outputs', 'test.png')
upscale = path.join('..', 'outputs', 'test-large.png')

print('upscaling test image...')
session = ort.InferenceSession(esrgan, providers=['DmlExecutionProvider'])

in_image = cv2.imread(output, cv2.IMREAD_UNCHANGED)

in_mat = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
print('shape before', np.shape(in_mat))
in_mat = np.transpose(in_mat, (2, 1, 0))[np.newaxis]
print('shape after', np.shape(in_mat))
in_mat = in_mat.astype(np.float32)
in_mat = in_mat/255

start_time = time.time()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
in_mat = torch.tensor(in_mat)
out_mat = session.run([output_name], {
    input_name: in_mat.cpu().numpy()
})[0]
elapsed_time = time.time() - start_time
print(elapsed_time)

print('output shape', np.shape(out_mat))
out_mat = np.squeeze(out_mat, (0))
print(np.shape(out_mat))
out_mat = np.transpose(out_mat, (2, 1, 0))
print(out_mat, np.shape(out_mat))
out_mat = np.clip(out_mat, 0.0, 1.0)
out_mat = out_mat * 255
out_mat = out_mat.astype(np.uint8)
out_image = cv2.cvtColor(out_mat, cv2.COLOR_RGB2BGR)

cv2.imwrite(upscale, out_image)

print('upscaled test image to %s' % upscale)
