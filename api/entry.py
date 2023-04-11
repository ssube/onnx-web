import multiprocessing
import threading
import waitress
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
        # convert the models
        from onnx_web.convert.__main__ import main as convert
        print("downloading and converting models to ONNX")
        convert()

        # create the server and load the config
        from onnx_web.main import main
        app, pool = main()

        # launch the image workers
        print("starting image workers")
        pool.start()

        # launch the API server
        print("starting API server")
        server = waitress.create_server(app, host="0.0.0.0", port=5000)
        thread = threading.Thread(target=server.run)
        thread.daemon = True
        thread.start()

        # launch the user's web browser
        print("opening web browser")
        url = "http://127.0.0.1:5000"
        webbrowser.open_new_tab(f"{url}?api={url}")

        # wait for enter and exit
        input("press enter to quit")
        server.close()
        thread.join(1.0)

        print("shutting down image workers")
        pool.join()
    except Exception as e:
        print(e)
