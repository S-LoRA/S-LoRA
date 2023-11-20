import filelock
import os
import json
from huggingface_hub import snapshot_download


def get_lock(model_name_or_path: str, cache_dir: str = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock

def hf_load_config(weights_dir, mode="adapter"):
    is_local = os.path.isdir(weights_dir)
    if not is_local:
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path=weights_dir):
            weights_dir = snapshot_download(weights_dir,
                                        allow_patterns=["*.bin", "*.json"])
    config_name = "adapter_config.json" if mode == "adapter" else "config.json"
    with open(os.path.join(weights_dir, config_name), "r") as f:
        return json.load(f), weights_dir

def hf_load_quantize_config(weights_dir):
    # As the weight_dir has already been updated in `hf_load_config` func in `_init_config` func, we assume that the weights_dir is local now.
    quantize_config_filename = None
    if os.path.isfile(os.path.join(weights_dir, "quantize_config.json")):
        quantize_config_filename = os.path.join(weights_dir, "quantize_config.json")
    elif os.path.isfile(os.path.join(weights_dir, "quant_config.json")):
        quantize_config_filename = os.path.join(weights_dir, "quant_config.json")
    else:
        raise ValueError(f"Can not find either the `quantize_config.json` file or the `quant_config.json` file under the weights_dir `{weights_dir}`. Please check if the model you are using is a quantified model!")

    with open(quantize_config_filename, "r") as f:
        return json.load(f)