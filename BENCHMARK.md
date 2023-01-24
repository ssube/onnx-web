# Very Rough Benchmarks

CUDA > ROCm > DirectML > drawing it yourself > CPU

Using 25 steps of Euler A in txt2img, 512x512.

- CPU:
  - AMD:
    - 5900HX: 7.5s/it, 150sec/image
    - 5950X: 4s/it, 100sec/image
    - 7950X: 3.5s/it, 90sec/image
- GPU:
  - AMD:
    - 6900XT
      - Win10, DirectML: 3.5it/s, 9sec/image
      - Ubuntu 20.04, ROCm 5.2: 4.5it/s, 6sec/image
  - Nvidia:
    - 4090: 6.5it/s, 4sec/image
