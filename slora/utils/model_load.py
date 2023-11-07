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
