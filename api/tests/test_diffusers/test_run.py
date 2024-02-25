import unittest
from multiprocessing import Queue, Value
from os import path

from PIL import Image

from onnx_web.diffusers.run import (
    run_blend_pipeline,
    run_img2img_pipeline,
    run_inpaint_pipeline,
    run_txt2img_pipeline,
    run_upscale_pipeline,
)
from onnx_web.image.mask_filter import mask_filter_none
from onnx_web.image.noise_source import noise_source_uniform
from onnx_web.params import (
    Border,
    HighresParams,
    ImageParams,
    RequestParams,
    Size,
    TileOrder,
    UpscaleParams,
)
from onnx_web.server.context import ServerContext
from onnx_web.worker.command import JobCommand
from onnx_web.worker.context import WorkerContext
from tests.helpers import (
    TEST_MODEL_DIFFUSION_SD15,
    TEST_MODEL_DIFFUSION_SD15_INPAINT,
    test_device,
    test_needs_models,
    test_worker,
)

TEST_PROMPT = "an astronaut eating a hamburger"
TEST_SCHEDULER = "ddim"


class TestTxt2ImgPipeline(unittest.TestCase):
    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_basic(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        device = test_device()
        worker = WorkerContext(
            "test",
            device,
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start(
            JobCommand(
                "test-txt2img-basic", "test", "test", run_txt2img_pipeline, [], {}
            )
        )

        params = RequestParams(
            device,
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            size=Size(256, 256),
            upscale=UpscaleParams("test"),
            highres=HighresParams(False, 1, 0, 0),
        )

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-basic_0.png"))

        with Image.open("../outputs/test-txt2img-basic.png") as output:
            self.assertEqual(output.size, (256, 256))
            # TODO: test contents of image

    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_batch(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        device = test_device()
        worker = WorkerContext(
            "test",
            device,
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start(
            JobCommand(
                "test-txt2img-batch", "test", "test", run_txt2img_pipeline, [], {}
            )
        )

        params = RequestParams(
            device,
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
                batch=2,
            ),
            size=Size(256, 256),
            upscale=UpscaleParams("test"),
            highres=HighresParams(False, 1, 0, 0),
        )

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-batch_0.png"))
        self.assertTrue(path.exists("../outputs/test-txt2img-batch_1.png"))

        with Image.open("../outputs/test-txt2img-batch_0.png") as output:
            self.assertEqual(output.size, (256, 256))
            # TODO: test contents of image

    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_highres(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        device = test_device()
        worker = WorkerContext(
            "test",
            device,
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start(
            JobCommand(
                "test-txt2img-highres", "test", "test", run_txt2img_pipeline, [], {}
            )
        )

        params = RequestParams(
            device,
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
                unet_tile=256,
            ),
            size=Size(256, 256),
            upscale=UpscaleParams("test", scale=2, outscale=2),
            highres=HighresParams(True, 2, 0, 0),
        )

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-highres_0.png"))
        with Image.open("../outputs/test-txt2img-highres_0.png") as output:
            self.assertEqual(output.size, (512, 512))

    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_highres_batch(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        device = test_device()
        worker = WorkerContext(
            "test",
            device,
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start(
            JobCommand(
                "test-txt2img-highres-batch",
                "test",
                "test",
                run_txt2img_pipeline,
                [],
                {},
            )
        )

        params = RequestParams(
            device,
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
                batch=2,
            ),
            size=Size(256, 256),
            upscale=UpscaleParams("test"),
            highres=HighresParams(True, 2, 0, 0),
        )

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-highres-batch_0.png"))
        self.assertTrue(path.exists("../outputs/test-txt2img-highres-batch_1.png"))

        with Image.open("../outputs/test-txt2img-highres-batch_0.png") as output:
            self.assertEqual(output.size, (512, 512))


class TestImg2ImgPipeline(unittest.TestCase):
    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_basic(self):
        worker = test_worker()
        worker.start(
            JobCommand("test-img2img", "test", "test", run_txt2img_pipeline, [], {})
        )

        source = Image.new("RGB", (64, 64), "black")
        params = RequestParams(
            test_device(),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "img2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            upscale=UpscaleParams("test"),
            highres=HighresParams(False, 1, 0, 0),
        )
        run_img2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
            source,
            1.0,
        )

        self.assertTrue(path.exists("../outputs/test-img2img_0.png"))


class TestInpaintPipeline(unittest.TestCase):
    @test_needs_models([TEST_MODEL_DIFFUSION_SD15_INPAINT])
    def test_basic_white(self):
        worker = test_worker()
        worker.start(
            JobCommand(
                "test-inpaint-white", "test", "test", run_txt2img_pipeline, [], {}
            )
        )

        source = Image.new("RGB", (64, 64), "black")
        mask = Image.new("RGB", (64, 64), "white")
        params = RequestParams(
            test_device(),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15_INPAINT,
                "inpaint",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
                unet_tile=64,
            ),
            size=Size(*source.size),
            upscale=UpscaleParams("test"),
            highres=HighresParams(False, 1, 0, 0),
        )

        run_inpaint_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
            source,
            mask,
            Border.even(0),
            noise_source_uniform,
            mask_filter_none,
            "white",
            TileOrder.spiral,
            False,
            0.0,
        )

        self.assertTrue(path.exists("../outputs/test-inpaint-white_0.png"))

    @test_needs_models([TEST_MODEL_DIFFUSION_SD15_INPAINT])
    def test_basic_black(self):
        worker = test_worker()
        worker.start(
            JobCommand(
                "test-inpaint-black", "test", "test", run_txt2img_pipeline, [], {}
            )
        )

        source = Image.new("RGB", (64, 64), "black")
        mask = Image.new("RGB", (64, 64), "black")
        params = RequestParams(
            test_device(),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15_INPAINT,
                "inpaint",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
                unet_tile=64,
            ),
            size=Size(*source.size),
            upscale=UpscaleParams("test"),
            highres=HighresParams(False, 1, 0, 0),
        )

        run_inpaint_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
            source,
            mask,
            Border.even(0),
            noise_source_uniform,
            mask_filter_none,
            "black",
            TileOrder.spiral,
            False,
            0.0,
        )

        self.assertTrue(path.exists("../outputs/test-inpaint-black_0.png"))


class TestUpscalePipeline(unittest.TestCase):
    @test_needs_models(["../models/upscaling-stable-diffusion-x4"])
    def test_basic(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        device = test_device()
        worker = WorkerContext(
            "test",
            device,
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start(
            JobCommand("test-upscale", "test", "test", run_upscale_pipeline, [], {})
        )

        source = Image.new("RGB", (64, 64), "black")
        params = RequestParams(
            device,
            ImageParams(
                "../models/upscaling-stable-diffusion-x4",
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            size=Size(256, 256),
            upscale=UpscaleParams("test"),
            highres=HighresParams(False, 1, 0, 0),
        )
        run_upscale_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
            source,
        )

        self.assertTrue(path.exists("../outputs/test-upscale_0.png"))


class TestBlendPipeline(unittest.TestCase):
    def test_basic(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        device = test_device()
        worker = WorkerContext(
            "test",
            device,
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start(
            JobCommand("test-blend", "test", "test", run_blend_pipeline, [], {})
        )

        source = Image.new("RGBA", (64, 64), "black")
        mask = Image.new("RGBA", (64, 64), "white")
        params = RequestParams(
            device,
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
                unet_tile=64,
            ),
            size=Size(64, 64),
            upscale=UpscaleParams("test"),
        )
        run_blend_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            params,
            [source, source],
            mask,
        )

        self.assertTrue(path.exists("../outputs/test-blend_0.png"))
