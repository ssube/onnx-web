import os

def script_method(fn, _rcb=None):
    return fn

def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj

import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script

import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        from onnx_web.main import main
        app, pool = main()
        print("starting workers")
        pool.start()
        print("starting flask")
        app.run("0.0.0.0", 5000, debug=False)
        input("press the any key")
        pool.join()
    except Exception as e:
        print(e)
    finally:
        os.system("pause")
