import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from typing import List, Tuple

import aiohttp

from exp_suite import BenchmarkConfig, get_all_suites, to_dict
from trace import generate_requests
sys.path.append("../bench_lora")
from launch_server import base_model, adapter_dirs
from slora.utils.metric import reward

GB = 1024 ** 3

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def get_peak_mem(server):
    url = server + "/get_peak_mem"
    response = requests.post(url)
    return response.json()["peak_mem"]


async def send_request(
    server: str,
    req_id: str,
    model_dir: str,
    adapter_dir: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    debug: bool,
    id: int,
) -> None:
    request_start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    headers = {"User-Agent": "Benchmark Client"}
    url = server + "/generate"
    data = {
        'model_dir': model_dir,
        'lora_dir': adapter_dir,
        'inputs': prompt,
        'parameters': {
            'do_sample': False,
            'ignore_eos': True,
            'max_new_tokens': output_len,
                # 'temperature': 0.1,
        }
    }


    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)
            print(data["lora_dir"])
            print(output)
            if "alpaca-lora-7b" in data["lora_dir"] and id==0:
                output_ref = b'I am here to help you create a plan that works for you and your unique needs. I am here to help you create a plan that works for you and your unique needs. I am here to help you create a plan that works for you and your unique needs. I am here to help you create a plan that works for you and your unique needs. I am here to help you create a plan that works for you and your unique needs. I am here to help'
                assert output['generated_text'][0].encode() == output_ref
            if "alpaca-lora-7b" in data["lora_dir"] and id==1:
                output_ref = b'France. It is located in the north-central part of the country, on the Seine River. Paris is the most populous city in France, with a population of 2.2 million people. It is also the most populous city in the European Union. Paris is the center of culture, art, fashion, and gastronomy. It is also a major tourist destination, with millions of visitors each year.\nParis is divided into 20'
                assert output['generated_text'][0].encode() == output_ref
            if "bactrian-x-llama-7b-lora" in data["lora_dir"] and id==0:
                output_ref = '\nBrown Suga Wellness is a company that understands that you want to live a fulfilled life. To do that you need mental clarity and a well-crafted plan. The problem is you don’t have time to evaluate hundreds of coaches and companies to piece together a wellness plan that bring you questionable results or don’t cater to your unique needs as a woman of color which makes you feel stuck, overwhelmed,'.encode()
                assert output['generated_text'][0].encode() == output_ref
            if "bactrian-x-llama-7b-lora" in data["lora_dir"] and id==1:
                output_ref = 'France and is one of the most visited cities in the world. It is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also a hub for fashion, art, and culture, and is home to some of the world’s best restaurants and cafes.\nThe city is divided into 20 arrondissements, each with its own unique character'.encode()
                assert output['generated_text'][0].encode() == output_ref
            
            break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    
    return (prompt_len, output_len, request_latency, first_token_latency)


async def benchmark(
    server: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    debug=False,
    id=1,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        await asyncio.sleep(start + req.req_time - time.time())
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.adapter_dir}")
        task = asyncio.create_task(send_request(server,
                                                req.req_id, req.model_dir, req.adapter_dir, req.prompt,
                                                req.prompt_len, req.output_len, debug, id))
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency


def get_adapter_dirs(num_adapters):
    ret = []
    for i in range(num_adapters // len(adapter_dirs) + 1):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    return ret


def run_exp(server, config, seed=42):
    print([(k, v) for k, v in zip(BenchmarkConfig._fields, config)])

    num_adapters, alpha, req_rate, cv, duration, input_range, output_range = config
    # assert duration >= 30
    adapter_dirs = get_adapter_dirs(num_adapters)
    adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(num_adapters)]

    # generate requests
    requests1 = generate_requests(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, adapter_dirs,
                                 seed=seed, id=0)
    requests2 = generate_requests(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, adapter_dirs,
                                 seed=seed, id=1)

    # benchmark
    
    _ = asyncio.run(benchmark(server, requests1, id=0))
    _ = asyncio.run(benchmark(server, requests2, id=1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    suites = get_all_suites()

    for config in tqdm(suites, desc="suites"):
        _ = run_exp(args.server, config, args.seed)

