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

    parser.add_argument("--num-adapter", type=int, default=10)
    parser.add_argument("--num-token", type=int, default=10000)

    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--bmm", action="store_true")
    parser.add_argument("--batch-num-adapters", type=int, default=None)
    parser.add_argument("--enable-abort", action="store_true")

    parser.add_argument("--fair-weights", type=int, default=[], action="append")
    parser.add_argument("--scheduler", type=str, default="vtc_fair")
    args = parser.parse_args()

    base_model = BASE_MODEL[args.model_setting]
    adapter_dirs = LORA_DIR[args.model_setting]

    if args.backend == "slora":
        cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
        cmd += f" --model {base_model}"
        cmd += f" --tokenizer_mode auto"

        num_iter = args.num_adapter // len(adapter_dirs) + 1
        for i in range(num_iter):
            for adapter_dir in adapter_dirs:
                cmd += f" --lora {adapter_dir}-{i}"
        for x in args.fair_weights:
            cmd += f" --fair-weights {x}"

        if args.dummy:
            cmd += " --dummy"
        cmd += " --swap"
        cmd += f" --scheduler {args.scheduler}"
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

    # print(cmd)
    os.system(cmd)
