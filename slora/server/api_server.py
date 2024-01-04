# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
import torch
import uvloop
import sys

from .build_prompt import build_prompt

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import argparse
import json
from http import HTTPStatus
import uuid
import multiprocessing as mp
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
from .sampling_params import SamplingParams
from .httpserver.manager import HttpServerManager
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process

from slora.utils.net_utils import alloc_can_use_network_port
from slora.common.configs.config import setting
from .api_models import (
    ChatCompletionRequest,
    UsageInfo,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
)

from slora.mprophet.measure import ModelProphet
from slora.mprophet.lora_stats import LoRAProphet


GB = 1024 ** 3
MB = 1024 ** 2

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

isFirst = True


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"

@app.post("/generate")
async def generate(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    if "req_id" in request_dict:
        request_id = request_dict["req_id"]
    else:
        request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = []
    count_output_tokens = 0
    tokens = []
    async for request_output, metadata, finished in results_generator:
        count_output_tokens += 1
        if finished == -1:
            return Response(status_code=499)
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await httpserver_manager.abort(request_id)
            return Response(status_code=499)
        final_output.append(request_output)
        if return_details:
            metadata["text"] = request_output
            tokens.append(metadata)

    assert final_output is not None
    ret = {
        "generated_text": ["".join(final_output)],
        "count_output_tokens": count_output_tokens,
    }
    if return_details:
        ret["tokens"] = tokens
    return Response(content=json.dumps(ret, ensure_ascii=False).encode("utf-8"))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    if "req_id" in request_dict:
        request_id = request_dict["req_id"]
    else:
        request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata, finished in results_generator:
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": request_output,
                    "logprob": metadata.get("logprob", None),
                    "special": False
                },
                "generated_text": None,
                "finished": finished,
                "details": None
            }

            yield ("data:" + json.dumps(ret, ensure_ascii=False) + f"\n\n").encode(
                "utf-8"
            )

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request
) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.n > 1:
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The n parameter currently only supports 1"
        )

    if request.function_call != "none":
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The function call feature is not supported"
        )

    created_time = int(time.time())
    prompt = await build_prompt(request)
    sampling_params = SamplingParams(
        do_sample=request.do_sample,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        max_new_tokens=request.max_tokens,
        stop_sequences=request.stop
    )
    sampling_params.verify()

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    if not request.stream:
        final_output = []
        prompt_tokens = -1
        completion_tokens = 0
        async for request_output, metadata in results_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await httpserver_manager.abort(request_id)
                return Response(status_code=499)
            completion_tokens += 1
            if prompt_tokens == -1:
                prompt_tokens = metadata["prompt_tokens"]
            final_output.append(request_output)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        chat_message = ChatMessage(role="assistant", content="".join(final_output))
        choice = ChatCompletionResponseChoice(index=0, message=chat_message)
        resp = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
        return resp

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata in results_generator:
            delta_message = DeltaMessage(role="assistant", content=request_output)

            stream_choice = ChatCompletionStreamResponseChoice(
                index=0, delta=delta_message
            )

            stream_resp = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + stream_resp.json(ensure_ascii=False) + f"\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )


def print_mem_stats(args):
    model_dir = args.model_dir
    model_name = args.model_dir.split("/")[-1]
    try:
        fake_model = ModelProphet(model_name, model_dir=model_dir)
    except:
        fake_model = ModelProphet(model_name)
    model_size = fake_model.get_model_size()
    print(f"{model_name}: {model_size / GB:.2f} GB")
    peak_working_memory = fake_model.get_peak_working_memory(
            bs=20, context_len=512, tiling_dim=512)
    print(f"peak working mem for (bs=20, seqlen=512): {peak_working_memory / GB:.2f} GB")
    peak_working_memory = fake_model.get_peak_working_memory(
            bs=100, context_len=512, tiling_dim=512)
    print(f"peak working mem for (bs=100, seqlen=512): {peak_working_memory / GB:.2f} GB")
 
    tot_lora_size = 0
    for lora_dir in args.lora_dirs:
        lora_name = lora_dir.split("/")[-1]
        if args.dummy:
            fake_model = LoRAProphet(lora_name, model_name)
            try:
                fake_model = LoRAProphet(lora_name, model_name)
            except NotImplementedError as e:
                fake_model = LoRAProphet(lora_name, model_name,
                                         adapter_dir=lora_dir,
                                         base_model_dir=model_dir)
        else:
            fake_model = LoRAProphet(lora_name, model_name,
                                     adapter_dir=lora_dir,
                                     base_model_dir=model_dir)
        lora_size = fake_model.get_adapter_size()
        tot_lora_size += lora_size
        # print(f"{lora_name}, {base_name}: {lora_size / GB:.3f} GB")
    print(f"all adapters ({len(args.lora_dirs)}) estimated size: {tot_lora_size / GB:.2f} GB")
    print(f"avg adapter estimated size: {tot_lora_size / len(args.lora_dirs) / MB:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--model_dir", type=str, default=None,
                        help="the model weight dir path, the app will load config, weights and tokenizer from this dir")
    parser.add_argument("--tokenizer_mode", type=str, default="slow",
                        help="""tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is good for debug and test, 
                        when you want to get best performance, try auto mode""")
    parser.add_argument("--max_total_token_num", type=int, default=6000,
                        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)")
    parser.add_argument("--batch_max_tokens", type=int, default=None,
                        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM")
    parser.add_argument("--eos_id", type=int, default=2,
                        help="eos stop token id")
    parser.add_argument("--running_max_req_size", type=int, default=1000,
                        help="the max size for forward requests in the same time")
    parser.add_argument("--tp", type=int, default=1,
                        help="model tp parral size, the default is 1")
    parser.add_argument("--max_req_input_len", type=int, default=2048,
                        help="the max value for req input tokens num")
    parser.add_argument("--max_req_total_len", type=int, default=2048 + 1024,
                        help="the max value for req_input_len + req_output_len")
    parser.add_argument("--nccl_port", type=int, default=28765,
                        help="the nccl_port to build a distributed environment for PyTorch")
    parser.add_argument("--mode", type=str, default=[], nargs='+',
                        help="Model mode: [int8kv] [int8weight | int4weight]")
    parser.add_argument("--trust_remote_code", action='store_true',
                        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.")
    parser.add_argument("--disable_log_stats", action='store_true',
                        help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10,
                        help="log stats interval in second.")

    ''' slora arguments '''
    parser.add_argument("--lora-dirs", type=str, default=[], action="append",
                        help="the adapter weight dirs associate with base model dir")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--pool-size-lora", type=int, default=0)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-size", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="slora")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch-num-adapters", type=int, default=None)
    parser.add_argument("--enable-abort", action="store_true")

    # debug parameters
    # do not use, does not rule out the swap over MemAllocator
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-kernel", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--bmm", action="store_true")
    ''' end of slora arguments '''

    args = parser.parse_args()

    assert args.max_req_input_len < args.max_req_total_len
    setting["max_req_total_len"] = args.max_req_total_len
    setting["nccl_port"] = args.nccl_port

    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens
    else:
        assert (
            args.batch_max_tokens >= args.max_req_total_len
        ), "batch_max_tokens must >= max_req_total_len"

    can_use_ports = alloc_can_use_network_port(
        num=3 + args.tp, used_nccl_port=args.nccl_port
    )
    router_port, detokenization_port, httpserver_port = can_use_ports[0:3]
    model_rpc_ports = can_use_ports[3:]

    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args.model_dir,
        args.tokenizer_mode,
        router_port=router_port,
        httpserver_port=httpserver_port,
        total_token_num=args.max_total_token_num,
        max_req_input_len=args.max_req_input_len,
        max_req_total_len=args.max_req_total_len,
        trust_remote_code=args.trust_remote_code,
        dummy=args.dummy,
    )
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)
    proc_router = mp.Process(
        target=start_router_process,
        args=(
            args,
            router_port,
            detokenization_port,
            model_rpc_ports,
            args.mode,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenization_process,
        args=(
            args,
            detokenization_port,
            httpserver_port,
            pipe_detoken_writer,
            args.trust_remote_code,
        ),
    )
    proc_detoken.start()

    # wait load model ready
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print(
            "router init state:",
            router_init_state,
            "detoken init state:",
            detoken_init_state,
        )
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    print_mem_stats(args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn'), # this code will not be ok for settings to fork to subprocess
    main()
