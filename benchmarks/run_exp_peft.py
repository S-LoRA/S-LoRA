"""
To run with real mode:
python run_exp_peft.py --backend slora --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp_peft.py --backend slora --suite a10g --breakdown  --mode synthetic
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
from tqdm import tqdm
from typing import List, Tuple

import aiohttp
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from exp_suite import BenchmarkConfig, get_all_suites, to_dict
from trace import generate_requests, dummy_prompt, get_real_requests
sys.path.append("../bench_lora")
from exp_suite import BASE_MODEL, LORA_DIR
from slora.utils.metric import reward, attainment_func
from run_exp import get_adapter_dirs

GB = 1024 ** 3

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def get_peak_mem(server):
    url = server + "/get_peak_mem"
    response = requests.post(url)
    return response.json()["peak_mem"]

# benchmark for huggingface peft
def benchmark(
    backend: str,
    server: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    loaded_base_model,
    loaded_tokenizer,
    debug,
) -> None:
    start = time.time()
    all_requests = copy.deepcopy(input_requests)
    first_token_time = [None] * len(all_requests)
    latency = [None] * len(all_requests)

    while len(all_requests) > 0:
        # check the current queue
        queue = []
        for req in all_requests:
            if req.req_time < time.time() - start:
                queue.append(req)
            # else: # last possible request 
            #    continue

        if len(queue) > 0:
            print(f"processing {queue}")
            # sort the queue of the same adapter 
            grouped_queue = {}
            for req in queue:
                key = req.adapter_dir
                if key in grouped_queue.keys():
                    grouped_queue[key].append(req)
                else:
                    grouped_queue[key] = [req]
            
            for adapter_dir, group_reqs in grouped_queue.items():
                # load the current adapter
                try:
                    lora_rank = int(adapter_dir.split("-")[4])
                    print(f"using rank {lora_rank}")
                    peft_config = LoraConfig(
                        task_type="CAUSAL_LM", inference_mode=True, r=lora_rank,
                    )
                    cur_model = get_peft_model(loaded_base_model, peft_config)
                except:
                    print("using base model only")
                    cur_model = loaded_base_model

                cur_model.eval()
                # peft easily OOM, manually set this max batch size 
                # main 
                max_batch_size = 6
                # tab3 llama 7b
                # max_batch_size = 50
                # tab3 llama 13b
                # max_batch_size = 35
                subgrouped_queue = [[]]
                for subgroup_req in group_reqs:
                    if len(subgrouped_queue[-1]) < max_batch_size:
                        subgrouped_queue[-1].append(subgroup_req)
                    else:
                        subgrouped_queue.append([subgroup_req])
                
                for reqs in subgrouped_queue:
                    # generate 
                    padded_prompt_length = max([r.prompt_len for r in reqs])
                    padded_output_length = max([r.output_len for r in reqs])
                    padded_prompt = [dummy_prompt(padded_prompt_length) for _ in range(len(reqs))]
                    # pad to the maximum length and do batching
                    input_ids = loaded_tokenizer(padded_prompt).input_ids
                    output_ids = list(input_ids)
                    past_key_values = out = None
                    active_reqs = copy.deepcopy(reqs)
                    for i in range(padded_output_length):
                        if i == 0:  # prefill
                            out = cur_model(torch.as_tensor(input_ids, device="cuda"), use_cache=True)
                            logits = out.logits
                            past_key_values = out.past_key_values
                            # update first token latency
                            for _req in reqs:
                                first_token_time[_req.req_id] = time.time() - (start + _req.req_time)
                        else:  # decoding
                            # print(f"decoding shape {torch.as_tensor([token],device='cuda').shape}")
                            if i % 100 == 0:
                                print(len(past_key_values), len(past_key_values[0]), past_key_values[0][0].shape, token)
                            out = cur_model(
                                input_ids=torch.as_tensor(
                                    token,
                                    device="cuda",
                                ),
                                use_cache=True,
                                past_key_values=past_key_values,
                            )
                            logits = out.logits
                            past_key_values = out.past_key_values
                        last_token_logits = logits[:, -1, :]
                        # greedy
                        _, indices = torch.topk(last_token_logits, 1)
                        indices = indices.tolist()
                        # print(f"indices: {indices}")
                        token = []
                        # print(output_ids)
                        index_in_kv_cache = 0
                        keep_idx = list(range(len(active_reqs)))
                        for j in range(len(reqs)):
                            if reqs[j].req_id not in [_req.req_id for _req in active_reqs]:
                                continue
                            token.append(indices.pop(0))
                            output_ids[j].append(token[-1][0])
                            # update finished requests
                            if (i + 1) == reqs[j].output_len: 
                                # finish_indices.append(j)
                                # kv: (32, 2) tuples of [bs, num_heads, seq_len, hdim]
                                for _req in active_reqs:
                                    if _req.req_id == reqs[j].req_id:
                                        active_reqs.remove(_req)
                                keep_idx.remove(index_in_kv_cache)
                                # remove the token just added so that it does not go into the next iteration
                                token = token[:-1]
                                # print(len(past_key_values), len(past_key_values[0]), past_key_values[0][0].shape)
                                print(f"request {reqs[j].req_id} has finished with time elapsed {time.time() - start}")
                                latency[reqs[j].req_id] = (reqs[j].prompt_len, reqs[j].output_len, time.time() - (start + reqs[j].req_time), first_token_time[reqs[j].req_id])
                            index_in_kv_cache += 1
                        past_key_values = tuple(
                                        [tuple([t[keep_idx, :] for t in tt])
                                        for tt in past_key_values]
                                    )
                        # print(token, output_ids)
                    # detokenize
                    loaded_tokenizer.batch_decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
        
            # finish, remove from all_requests
            all_requests = all_requests[len(queue):]
            print(f"all_requests len {len(all_requests)} time elapsed: {time.time() - start}")
    print(latency)
    return latency


def get_res_stats(per_req_latency, benchmark_time, backend, warmup_time=0, warmup_num=0):
    # get throughput
    throughput = len(per_req_latency) / benchmark_time
    # if backend == "slora":
    #     peak_mem = get_peak_mem(server)
    #     print(f"GPU peak memory (GB):", [[f"{x / GB:.2f}" for x in tpg] for tpg in peak_mem])
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {throughput:.2f} requests/s")

    # print(f"debug: {len(per_req_latency)} {warmup_num * 2} {benchmark_time}, {warmup_time * 2}")
    strip_throughput = (len(per_req_latency) - warmup_num * 2) / (benchmark_time - warmup_time * 2)
    print(f"Throughput strip: {strip_throughput:.2f} requests/s")

    # compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _ in per_req_latency])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency, _ in per_req_latency
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency, _ in per_req_latency
    ])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")

    # compute the first token latency
    first_token_latency = [latency for _, _, _, latency in per_req_latency]
    avg_first_token_latency = np.mean(first_token_latency)
    print(f"Average first token latency: {avg_first_token_latency:.2f} s")
    print(f"90 percentile first token latency: < {np.percentile(first_token_latency, 90):.2f} s")
    print(f"50 percentile first token latency: < {np.percentile(first_token_latency, 50):.2f} s")
    satisfaction = [reward(latency) for _, _, _, latency in per_req_latency]
    avg_satisfaction = np.mean(satisfaction)
    print(f"Average satisfaction: {avg_satisfaction:.2f}")
    print(f"90 percentile satisfaction: > {np.percentile(satisfaction, 10):.2f}")
    print(f"50 percentile satisfaction: > {np.percentile(satisfaction, 50):.2f}")

    attainment = [attainment_func(latency) for _, _, _, latency in per_req_latency]
    avg_attainment = np.mean(attainment)
    print(f"Average attainment: {avg_attainment:.2f}")

    # dump results
    if backend == "slora":
        # TODO
        # single_gpu_peak_mem = peak_mem
        single_gpu_peak_mem = 0
    else:
        single_gpu_peak_mem = 0

    result = {"total_time": benchmark_time, "gpu_peak_mem": single_gpu_peak_mem,
              "throughput": throughput, "strip_throughput": strip_throughput,
              "avg_latency": avg_latency, "avg_per_token_latency": avg_per_token_latency,
              "avg_per_output_token_latency": avg_per_output_token_latency,
              "avg_first_token_latency": avg_first_token_latency,
              "avg_satisfaction": avg_satisfaction,
              "avg_attainment": avg_attainment}
    res = {"config": to_dict(config), "result": result}
    
    return res

def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""

    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory

def run_exp(model_setting, backend, server, config, output, num_gpus, mode, seed=42, debug=False):
    if mode == "real":
        print("*** num_adapters, cv and alpha are not used in real mode ***")
    print([(k, v) for k, v in zip(BenchmarkConfig._fields, config)])

    num_adapters, alpha, req_rate, cv, duration, input_range, output_range = config
    # assert duration >= 30
    if mode == "synthetic":
        base_model = BASE_MODEL[model_setting]
        adapter_dirs = LORA_DIR[model_setting]
        adapter_dirs = get_adapter_dirs(num_adapters, adapter_dirs)
        adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(num_adapters)]
        if num_adapters == 0:
             adapter_dirs = [(base_model, None)]
             num_adapters = 1
        requests = generate_requests(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, adapter_dirs,
                                 seed=seed)
    else:
        # first generate your data using real_trace/clean_chat_data.py
        from launch_server import adapter_dirs
        adapter_dirs, requests = get_real_requests(trace_file="real_trace/clean_chat_conv_20231016.json", req_rate=req_rate, duration=duration,
                                      adapter_dirs=adapter_dirs, input_range=input_range, output_range=output_range, seed=seed)
        print(requests)
    
    # load base model outside so it does not count time
    print(base_model)
    
    kwargs = {"torch_dtype": torch.bfloat16}
    kwargs["device_map"] = "sequential"
    available_gpu_memory = get_gpu_memory(num_gpus)
    kwargs["max_memory"] = {
        i: str(int(available_gpu_memory[i] * 0.15)) + "GiB"
        for i in range(num_gpus)
    }
    loaded_base_model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    loaded_tokenizer = AutoTokenizer.from_pretrained(base_model)
    #for r in requests:
    #    print(r.adapter_dir)
    
    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    # benchmark
    benchmark_start_time = time.time()
    per_req_latency = benchmark(backend, server, requests, loaded_base_model, loaded_tokenizer, debug)
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    warmup_time = 10
    warmup_num = int(req_rate * warmup_time)
    res = get_res_stats(per_req_latency, benchmark_time, backend,
                        warmup_time=warmup_time, warmup_num=warmup_num)

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="default", required=True)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-copy", action="store_true")

    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--model-setting", type=str, default="S1")
    args = parser.parse_args()

    assert not args.no_lora_copy or args.no_lora_compute
    assert not (args.debug and args.breakdown)
    
    args.backend = "peft"
    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.mode}_" + args.backend + ".jsonl"

    suites = get_all_suites(mode=args.mode, debug=args.debug, suite=args.suite, breakdown=args.breakdown)

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
                            args.output, args.num_gpus, args.mode, args.seed, args.debug)

