import argparse
import os

# base_model = "dummy-llama-7b"
base_model = "huggyllama/llama-7b"
adapter_dirs = ["tloen/alpaca-lora-7b", "MBZUAI/bactrian-x-llama-7b-lora"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-adapter", type=int)
    parser.add_argument("--num-token", type=int)
    parser.add_argument("--pool-size-lora", type=int)

    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    args = parser.parse_args()

    
    if args.num_adapter is None: args.num_adapter = 4
    if args.num_token is None: args.num_token = 10000
    if args.pool_size_lora is None: args.pool_size_lora = 0
 
    cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
    cmd += f" --model {base_model}"
    cmd += f" --tokenizer_mode auto"
    cmd += f" --pool-size-lora {args.pool_size_lora}"

    num_iter = args.num_adapter // len(adapter_dirs) + 1
    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            cmd += f" --lora {adapter_dir}-{i}"

    cmd += " --swap"
    # cmd += " --scheduler pets"
    # cmd += " --profile"
    if args.no_lora_compute:
        cmd += " --no-lora-compute"
    if args.no_prefetch:
        cmd += " --prefetch False"
    if args.no_mem_pool:
        cmd += " --no-mem-pool"

    os.system(cmd)
