from logging import getLogger

from ..params import ImageParams, Size
from ..server import ServerContext
from ..worker import WorkerContext

logger = getLogger(__name__)


def run_txt2txt_pipeline(
    job: WorkerContext,
    _server: ServerContext,
    params: ImageParams,
    _size: Size,
    output: str,
) -> None:
    from transformers import AutoTokenizer, GPTJForCausalLM

    # tested with "EleutherAI/gpt-j-6B"
    model = "EleutherAI/gpt-j-6B"
    tokens = 1024

    device = job.get_device()

    pipe = GPTJForCausalLM.from_pretrained(model).to(device.torch_str())
    tokenizer = AutoTokenizer.from_pretrained(model)

    input_ids = tokenizer.encode(params.prompt, return_tensors="pt").to(
        device.torch_str()
    )
    results = pipe.generate(
        input_ids,
        do_sample=True,
        max_length=tokens,
        temperature=0.8,
    )
    result_text = tokenizer.decode(results[0], skip_special_tokens=True)

    print("Server says: %s" % result_text)

    logger.info("finished txt2txt job: %s", output)
