import os
import json
import torch

from dancingmodel.models.internlm.layer_infer.transformer_layer_infer import InternlmTransformerLayerInfer
from dancingmodel.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight

from dancingmodel.common.mem_manager import MemoryManager
from dancingmodel.models.llama.model import LlamaTpPartModel


class InternlmTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = InternlmTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = InternlmTransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
    

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return 
    
    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(self.max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_attention_heads"] // self.world_size_,
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])
        return