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
    parser.add_argument("--num-token", type=int, default=8000)

    parser.add_argument("--scheduler", type=str, default="vtc_fair")
    parser.add_argument("--fair-weights", type=int, default=[], action="append")
    args = parser.parse_args()
    # TODO remove lora

    base_model = BASE_MODEL[args.model_setting]
    adapter_dirs = LORA_DIR[args.model_setting]

    if args.backend == "slora":
        cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
        cmd += f" --model {base_model}"
        cmd += f" --tokenizer_mode auto"
        cmd += " --dummy"
        cmd += " --swap"
        cmd += f" --scheduler {args.scheduler}"

        num_iter = args.num_adapter // len(adapter_dirs) + 1
        for i in range(num_iter):
            for adapter_dir in adapter_dirs:
                cmd += f" --lora {adapter_dir}-{i}"
        for x in args.fair_weights:
            cmd += f" --fair-weights {x}"

    # print(cmd)
    os.system(cmd)
