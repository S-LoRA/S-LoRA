"""
To run with real mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode synthetic
default to synthetic mode.
"""
import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import List, Tuple

import aiohttp

from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import generate_requests, get_real_requests
sys.path.append("../bench_lora")
from slora.utils.metric import reward, attainment_func

GB = 1024 ** 3


@dataclass
class Response:
    adapter_dir: str
    prompt_len: int
    output_len: int
    request_latency: float
    first_token_latency: float
    req_time: float


async def send_request(
    backend: str,
    server: str,
    req_time: float,
    req_id: str,
    model_dir: str,
    adapter_dir: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    debug: bool,
) -> None:
    request_start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    headers = {"User-Agent": "Benchmark Client"}
    url = server + "/generate_stream"
    
    if backend in ["slora"]:
        data = {
            'model_dir': model_dir,
            'lora_dir': adapter_dir,
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            },
            'req_id': req_id,
        }
    else:
        raise NotImplementedError()

    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            # output = json.loads(output)
            # print(output)
            
            if '\"finished\": -1' not in output:
                break
            else:
                first_token_latency = None
                break
            # #     print(output)
            # #     print(json.loads(output))
            # break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    print(f"req_id {req_id} req_time {req_time} adapter_dir {adapter_dir} "
          f"prompt_len {prompt_len} output_len {output_len} "
          f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s")
    return Response(adapter_dir, prompt_len, output_len,
                    request_latency, first_token_latency, req_time)


async def benchmark(
    backend: str,
    server: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    debug=False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        await asyncio.sleep(start + req.req_time - time.time())
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.adapter_dir}")
        task = asyncio.create_task(send_request(backend, server, req.req_time,
                                                req.req_id, req.model_dir, req.adapter_dir,
                                                req.prompt, req.prompt_len, req.output_len,
                                                debug))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    return responses


def get_adapter_dirs(num_adapters, adapter_dirs, backend=None):
    ret = []
    num_iter = num_adapters // len(adapter_dirs) + 1

    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    return ret

def get_res_stats(responses, benchmark_time, backend):
    # get throughput
    num_abort = len([x for x in responses if x.first_token_latency is None])
    responses = [x for x in responses if x.first_token_latency is not None]
    throughput = len(responses) / benchmark_time
    # print(responses)
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Aborted Request: {num_abort}")
    print(f"Throughput: {throughput:.2f} requests/s")

    # compute the latency statistics.
    avg_latency = np.mean([x.request_latency for x in responses])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        x.request_latency / (x.prompt_len + x.output_len)
        for x in responses
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean([
        x.request_latency / x.output_len
        for x in responses
    ])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")

    # compute the first token latency
    first_token_latency = [x.request_latency for x in responses]
    avg_first_token_latency = np.mean(first_token_latency)
    print(f"Average first token latency: {avg_first_token_latency:.2f} s")
    print(f"90 percentile first token latency: < {np.percentile(first_token_latency, 90):.2f} s")
    print(f"50 percentile first token latency: < {np.percentile(first_token_latency, 50):.2f} s")

    # compute the attainment
    attainment = [attainment_func(x.request_latency) for x in responses]
    avg_attainment = np.mean(attainment)
    print(f"Average attainment: {avg_attainment:.2f}")

    # compute the per adapter first token latency (ftl) and attainment
    ftl_per_adapter = {}
    attainment_per_adapter = {}
    for x in responses:
        if x.adapter_dir not in ftl_per_adapter:
            ftl_per_adapter[x.adapter_dir] = [x.request_latency]
            attainment_per_adapter[x.adapter_dir] = [attainment_func(x.request_latency)]
        else:
            ftl_per_adapter[x.adapter_dir].append(x.request_latency)
            attainment_per_adapter[x.adapter_dir].append(attainment_func(x.request_latency))
    for k, v in ftl_per_adapter.items():
        print(f"Average first token latency ({len(v)}) for adapter {k}: {np.mean(v)} s")
    for k, v in attainment_per_adapter.items():
        print(f"Average attainment ({len(v)}) for adapter {k}: {np.mean(v)}")

    # dump results
    result = {"total_time": benchmark_time, "num_abort": num_abort,
              "throughput": throughput,
              "avg_latency": avg_latency, "avg_per_token_latency": avg_per_token_latency,
              "avg_per_output_token_latency": avg_per_output_token_latency,
              "avg_first_token_latency": avg_first_token_latency,
              "avg_attainment": avg_attainment,
              "responses": [asdict(x) for x in responses]}
    res = {"config": to_dict(config), "result": result}
    
    return res


def run_exp(model_setting, backend, server, config, output, seed=42, debug=False):
    print([(k, v) for k, v in zip(BenchmarkConfig._fields, config)])

    num_adapters, alpha, req_rate, cv, duration, input_range, output_range, on_off, mode = config
    # assert duration >= 30
    base_model = BASE_MODEL[model_setting]
    adapter_dirs = LORA_DIR[model_setting]
    adapter_dirs = get_adapter_dirs(num_adapters, adapter_dirs)
    adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(num_adapters)]
    if num_adapters == 0:
        adapter_dirs = [(base_model, None)]
        num_adapters = 1
    requests = generate_requests(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, on_off, mode, adapter_dirs,
                                 seed=seed)
    avg_prompt_len = np.mean([req.prompt_len for req in requests])
    avg_output_len = np.mean([req.output_len for req in requests])
    avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
    print("avg_len:", avg_len, "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len)
       
    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    # benchmark
    benchmark_start_time = time.time()
    responses = asyncio.run(benchmark(backend, server, requests, debug))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    res = get_res_stats(responses, benchmark_time, backend)

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="slora")
    parser.add_argument("--suite", type=str, default="default")

    parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.suite}.jsonl"
    if args.debug:
        args.output = "debug_" + args.output

    suites = get_all_suites(debug=args.debug, suite=args.suite)

    if not args.append:
        os.system(f"rm {args.output}")
        results = []
    else:
        with open(args.output, "r") as f:
            lines = f.readlines()
        results = [json.loads(line)["config"] for line in lines]

    for config in tqdm(suites, desc="suites"):
        if to_dict(config) not in results:
            stats = run_exp(args.model_setting, args.backend, args.server, config,
                            args.output, args.seed, args.debug)
