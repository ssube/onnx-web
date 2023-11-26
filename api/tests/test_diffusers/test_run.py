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
    Size,
    TileOrder,
    UpscaleParams,
)
from onnx_web.server.context import ServerContext
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

        worker = WorkerContext(
            "test",
            test_device(),
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start("test")

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            Size(256, 256),
            ["test-txt2img-basic.png"],
            UpscaleParams("test"),
            HighresParams(False, 1, 0, 0),
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-basic.png"))
        output = Image.open("../outputs/test-txt2img-basic.png")
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

        worker = WorkerContext(
            "test",
            test_device(),
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start("test")

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
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
            Size(256, 256),
            ["test-txt2img-batch-0.png", "test-txt2img-batch-1.png"],
            UpscaleParams("test"),
            HighresParams(False, 1, 0, 0),
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-batch-0.png"))
        self.assertTrue(path.exists("../outputs/test-txt2img-batch-1.png"))

        output = Image.open("../outputs/test-txt2img-batch-0.png")
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

        worker = WorkerContext(
            "test",
            test_device(),
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start("test")

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
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
            Size(256, 256),
            ["test-txt2img-highres.png"],
            UpscaleParams("test", scale=2, outscale=2),
            HighresParams(True, 2, 0, 0),
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-highres.png"))
        output = Image.open("../outputs/test-txt2img-highres.png")
        self.assertEqual(output.size, (512, 512))

    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_highres_batch(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        worker = WorkerContext(
            "test",
            test_device(),
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start("test")

        run_txt2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
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
            Size(256, 256),
            ["test-txt2img-highres-batch-0.png", "test-txt2img-highres-batch-1.png"],
            UpscaleParams("test"),
            HighresParams(True, 2, 0, 0),
        )

        self.assertTrue(path.exists("../outputs/test-txt2img-highres-batch-0.png"))
        self.assertTrue(path.exists("../outputs/test-txt2img-highres-batch-1.png"))

        output = Image.open("../outputs/test-txt2img-highres-batch-0.png")
        self.assertEqual(output.size, (512, 512))


class TestImg2ImgPipeline(unittest.TestCase):
    @test_needs_models([TEST_MODEL_DIFFUSION_SD15])
    def test_basic(self):
        worker = test_worker()
        worker.start("test")

        source = Image.new("RGB", (64, 64), "black")
        run_img2img_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            ["test-img2img.png"],
            UpscaleParams("test"),
            HighresParams(False, 1, 0, 0),
            source,
            1.0,
        )

        self.assertTrue(path.exists("../outputs/test-img2img.png"))


class TestInpaintPipeline(unittest.TestCase):
    @test_needs_models([TEST_MODEL_DIFFUSION_SD15_INPAINT])
    def test_basic_white(self):
        worker = test_worker()
        worker.start("test")

        source = Image.new("RGB", (64, 64), "black")
        mask = Image.new("RGB", (64, 64), "white")
        run_inpaint_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15_INPAINT,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            Size(*source.size),
            ["test-inpaint-white.png"],
            UpscaleParams("test"),
            HighresParams(False, 1, 0, 0),
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

        self.assertTrue(path.exists("../outputs/test-inpaint-white.png"))

    @test_needs_models([TEST_MODEL_DIFFUSION_SD15_INPAINT])
    def test_basic_black(self):
        worker = test_worker()
        worker.start("test")

        source = Image.new("RGB", (64, 64), "black")
        mask = Image.new("RGB", (64, 64), "black")
        run_inpaint_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15_INPAINT,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            Size(*source.size),
            ["test-inpaint-black.png"],
            UpscaleParams("test"),
            HighresParams(False, 1, 0, 0),
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

        self.assertTrue(path.exists("../outputs/test-inpaint-black.png"))


class TestUpscalePipeline(unittest.TestCase):
    @test_needs_models(["../models/upscaling-stable-diffusion-x4"])
    def test_basic(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        worker = WorkerContext(
            "test",
            test_device(),
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start("test")

        source = Image.new("RGB", (64, 64), "black")
        run_upscale_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            ImageParams(
                "../models/upscaling-stable-diffusion-x4",
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            Size(256, 256),
            ["test-upscale.png"],
            UpscaleParams("test"),
            HighresParams(False, 1, 0, 0),
            source,
        )

        self.assertTrue(path.exists("../outputs/test-upscale.png"))


class TestBlendPipeline(unittest.TestCase):
    def test_basic(self):
        cancel = Value("L", 0)
        logs = Queue()
        pending = Queue()
        progress = Queue()
        active = Value("L", 0)
        idle = Value("L", 0)

        worker = WorkerContext(
            "test",
            test_device(),
            cancel,
            logs,
            pending,
            progress,
            active,
            idle,
            3,
            0.1,
        )
        worker.start("test")

        source = Image.new("RGBA", (64, 64), "black")
        mask = Image.new("RGBA", (64, 64), "white")
        run_blend_pipeline(
            worker,
            ServerContext(model_path="../models", output_path="../outputs"),
            ImageParams(
                TEST_MODEL_DIFFUSION_SD15,
                "txt2img",
                TEST_SCHEDULER,
                TEST_PROMPT,
                3.0,
                1,
                1,
            ),
            Size(64, 64),
            ["test-blend.png"],
            UpscaleParams("test"),
            [source, source],
            mask,
        )

        self.assertTrue(path.exists("../outputs/test-blend.png"))
