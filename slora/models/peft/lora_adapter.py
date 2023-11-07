import re
import torch
import os

from slora.mprophet.lora_config import get_lora_config_json

from slora.models.peft.layer_weights.hf_load_utils import load_hf_weights
from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
from slora.utils.model_load import hf_load_config


def get_lora_config(lora_dir, dummy):
    if dummy:
        return get_lora_config_json(lora_dir), lora_dir
    else:
        lora_dir = re.sub(r'-(\d+)$', '', lora_dir)
        return hf_load_config(lora_dir)


class LoraTpPartAdapter:

    def __init__(self, tp_rank, world_size, lora_dir, network_config,
                 swap=False, dummy=False, no_lora_swap=False, prefetch_stream=None):
        assert world_size == 1
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.lora_dir = lora_dir
        self.network_config = network_config

        self.lora_config, lora_dir = get_lora_config(lora_dir, dummy)

        self.r = self.lora_config["r"]
        self.lora_alpha = self.lora_config["lora_alpha"]
        self.scaling = self.lora_alpha / self.r
 
        self.layers = [
            LoraLayerWeight(i, tp_rank, world_size, self.lora_config, network_config, torch.float16,
                            no_lora_swap=no_lora_swap, prefetch_stream=prefetch_stream)
            for i in range(network_config["num_hidden_layers"])
        ]

        self.prefetch_stream = prefetch_stream

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

