import filelock
import json
import re
import torch
import os

from slora.mprophet.lora_config import get_lora_config_json

from slora.models.peft.layer_weights.hf_load_utils import load_hf_weights
from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
from huggingface_hub import snapshot_download


def get_lock(model_name_or_path: str, cache_dir: str = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock


def get_lora_config(lora_dir, dummy):
    if dummy:
        return get_lora_config_json(lora_dir)
    else:
        lora_dir = re.sub(r'-(\d+)$', '', lora_dir)
        is_local = os.path.isdir(lora_dir)
        if not is_local:
            # Use file lock to prevent multiple processes from
            # downloading the same model weights at the same time.
            with get_lock(model_name_or_path=lora_dir):
                lora_dir = snapshot_download(lora_dir,
                                            allow_patterns=["*.bin", "*.json"])
        with open(os.path.join(lora_dir, "adapter_config.json"), "r") as f:
            return json.load(f)


class LoraTpPartAdapter:

    def __init__(self, tp_rank, world_size, lora_dir, network_config,
                 swap=False, dummy=False, no_lora_swap=False, prefetch_stream=None):
        assert world_size == 1
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.lora_dir = lora_dir
        self.network_config = network_config

        self.lora_config = get_lora_config(lora_dir, dummy)

        self.r = self.lora_config["r"]
        self.lora_alpha = self.lora_config["lora_alpha"]
        self.scaling = self.lora_alpha / self.r
 
        self.layers = [
            LoraLayerWeight(i, tp_rank, world_size, self.lora_config, network_config, torch.float16,
                            no_lora_swap=no_lora_swap, prefetch_stream=prefetch_stream)
            for i in range(network_config["num_hidden_layers"])
        ]

        self.prefetch_stream = prefetch_stream
        if not dummy:
            lora_dir = re.sub(r'-(\d+)$', '', lora_dir)
            is_local = os.path.isdir(lora_dir)
            if not is_local:
                # Use file lock to prevent multiple processes from
                # downloading the same model weights at the same time.
                with get_lock(model_name_or_path=lora_dir):
                    lora_dir = snapshot_download(lora_dir,
                                                allow_patterns=["*.bin", "*.json"])

        load_hf_weights("fp16", lora_dir, transformer_layer_list=self.layers,
                        swap=swap, dummy=dummy)


    def is_on_gpu(self,):
        return (self.layers[0].w_combined is not None)


    def load_to_gpu(self, prefetch=False, bmm=False):
        if prefetch:
            with self.prefetch_stream:
                for layer_weight in self.layers:
                    layer_weight.load_to_gpu(bmm=bmm)
        else:
            for layer_weight in self.layers:
                layer_weight.load_to_gpu(bmm=bmm)


    def offload_from_gpu(self,):
        for layer_weight in self.layers:
            layer_weight.offload_from_gpu()

