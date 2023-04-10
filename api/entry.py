import multiprocessing
import os
import webbrowser

def script_method(fn, _rcb=None):
    return fn

def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj

import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        from onnx_web.convert.__main__ import main as convert
        print("converting models to ONNX")
        convert()

        from onnx_web.main import main
        app, pool = main()
        print("starting image workers")
        pool.start()

        print("starting API server")
        app.run("0.0.0.0", 5000, debug=False)

        url = "http://127.0.0.1:5000"
        webbrowser.open_new_tab(f"{url}?api={url}")

        input("press enter to quit")
        pool.join()
    except Exception as e:
        print(e)
    finally:
        os.system("pause")
