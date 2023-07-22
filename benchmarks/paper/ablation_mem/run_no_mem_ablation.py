import argparse
import os
import signal
import subprocess
import time

from exp_suite import BASE_MODEL, LORA_DIR


NUM_ADAPTER = {
    "a10g": {
        "S1": 200,
        "S2": 200,
        "S3": 200,
        "S4": 200,
    }
}

MEM_SIZE = {
    "a10g": {
        "S1": 15000,
        "S2": 15000,
        "S3": 10000,
        "S4": 10000,
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
 
    # for setting in ["S1", "S2", "S3", "S4"]:
    for setting in ["S2"]:
        # for option in ["", "--no-mem-pool"]:
        for option in [""]:
            print(f"run for {setting} and option {option}")
            num_adapter = NUM_ADAPTER[args.device][setting]
            num_token = MEM_SIZE[args.device][setting]
            # launch_cmd = f"python ../launch_server.py " \
            #              f" --device {args.device} " \
            #              f" --model-setting {setting} " \
            #              f" --backend slora " \
            #              f" --num-adapter {num_adapter} " \
            #              f" --num-token {num_token} "
            # launch_cmd += option
            # os.system(launch_cmd)

            # launch_cmd = ["python", "../launch_server.py"]
            # launch_cmd += ["--device", f"{args.device}"]
            # launch_cmd += ["--model-setting", f"{setting}"]
            # launch_cmd += ["--backend", "slora"]
            # launch_cmd += ["--num-adapter", f"{num_adapter}"]
            # launch_cmd += ["--num-token", f"{num_token}"]
            # if option != "":
            #     launch_cmd += [option]

            base_model = BASE_MODEL[setting]
            adapter_dirs = LORA_DIR[setting]
            launch_cmd = ["python", "-m", "slora.server.api_server"]
            launch_cmd += ["--max_total_token_num", f"{num_token}"]
            launch_cmd += ["--model", f"{base_model}"]
            launch_cmd += ["--tokenizer_mode", "auto"]
            num_iter = num_adapter // len(adapter_dirs) + 1
            for i in range(num_iter):
                for adapter_dir in adapter_dirs:
                    launch_cmd += ["--lora", f"{adapter_dir}-{i}"]
            launch_cmd += ["--dummy", "--swap"]
            if option == "--no-mem-pool":
                launch_cmd += "--no-mem-pool"

            # print(launch_cmd)
            # launch_server = subprocess.Popen(launch_cmd, start_new_session=True)
            # launch_server = subprocess.Popen(launch_cmd)
            # time.sleep(30)

            filename = f"ablation_mem_{args.device}_{setting}"
            if option == "--no-mem-pool":
                filename += f"_no_mem.jsonl"
            else:
                filename += f".jsonl"
            # run_cmd = f"python ../run_exp.py " \
            #           f" --backend slora " \
            #           f" --suite ablation-no-mem " \
            #           f" --model-setting {setting} " \
            #           f" --mode synthetic " \
            #           f" --output {filename} "
            # os.system(run_cmd)
            run_cmd = ["python", "../run_exp.py"]
            run_cmd += ["--backend", "slora"]
            run_cmd += ["--suite", "ablation-no-mem"]
            run_cmd += ["--model-setting", f"{setting}"]
            run_cmd += ["--mode", "synthetic"]
            run_cmd += ["--output", f"{filename}"]
            run_cmd += ["--debug"]
            print(run_cmd)
            run_exp = subprocess.Popen(run_cmd)

            while run_exp.poll() is None:
                pass
            launch_server.terminate()
            launch_server.kill()
            # os.killpg(launch_server.pid, signal.SIGTERM)
            # os.killpg(launch_server.pid, signal.SIGKILL)

