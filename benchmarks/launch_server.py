import argparse
import os
import psutil
import sys

from exp_suite import BASE_MODEL, LORA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="debug")
    parser.add_argument("--backend", type=str, default="slora",
                        choices=["slora", "vllm", "lightllm", "vllm-packed"])
    parser.add_argument("--model-setting", type=str, default="S1")

    parser.add_argument("--num-adapter", type=int)
    parser.add_argument("--num-token", type=int)

    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--bmm", action="store_true")
    parser.add_argument("--batch-num-adapters", type=int, default=None)
    parser.add_argument("--enable-abort", action="store_true")
    parser.add_argument("--vllm-mem-ratio", type=float, default=0.95)
    args = parser.parse_args()

    base_model = BASE_MODEL[args.model_setting]
    adapter_dirs = LORA_DIR[args.model_setting]

    if args.device == "a10g":
        if args.num_adapter is None: args.num_adapter = 200
        if args.num_token is None: args.num_token = 14000
    elif args.device == "h100":
        if args.num_adapter is None: args.num_adapter = 1000
        if args.num_token is None: args.num_token = 120000
    elif args.device == "debug":
        if args.num_adapter is None: args.num_adapter = 30
        if args.num_token is None: args.num_token = 14000
        if args.no_mem_pool:
            args.num_token -= 64 * 4 * 18
    

    if args.backend == "slora":
        cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
        cmd += f" --model {base_model}"
        cmd += f" --tokenizer_mode auto"

        num_iter = args.num_adapter // len(adapter_dirs) + 1
        for i in range(num_iter):
            for adapter_dir in adapter_dirs:
                cmd += f" --lora {adapter_dir}-{i}"

        cmd += " --dummy"
        cmd += " --swap"
        # cmd += " --scheduler pets"
        # cmd += " --profile"
        if args.enable_abort:
            cmd += " --enable-abort"
        if args.batch_num_adapters:
            cmd += f" --batch-num-adapters {args.batch_num_adapters}"
        if args.no_lora_compute:
            cmd += " --no-lora-compute"
        if args.prefetch:
            cmd += " --prefetch"
        if args.no_mem_pool:
            cmd += " --no-mem-pool"
        # cmd += " --no-lora-copy"
        # cmd += " --no-kernel"
        if args.bmm:
            cmd += " --bmm"

    elif args.backend == "lightllm":
        cmd = f"python -m lightllm.server.api_server" \
              f" --model_dir {base_model} --tp 1 --max_total_token_num {args.num_token}" \
              f" --tokenizer_mode auto" \
              f" --host 127.0.0.1 --port 8000"

    elif args.backend == "vllm":
        cmd = f"python -m vllm.entrypoints.api_server" \
              f" --model {base_model} --swap-space 16" \
              f" --disable-log-requests" \
              f" --host 127.0.0.1 --port 8000"

    elif args.backend == "vllm-packed":
        gpu_memory = args.vllm_mem_ratio / (args.num_adapter)

        for i in range(args.num_adapter):
            pid = os.fork()
            if pid == 0:
                cmd = f"python -m vllm.entrypoints.api_server" \
                    f" --model {base_model} --swap-space 0" \
                    f" --gpu-memory-utilization {gpu_memory}" \
                    f" --disable-log-requests" \
                    f" --host 127.0.0.1 --port {8000 + i}"
                os.system(cmd)
                sys.exit(0)

        for _ in range(args.num_adapter):
            os.wait()

        sys.exit(0)

    # print(cmd)
    os.system(cmd)
