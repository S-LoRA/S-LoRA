import numpy as np
import torch
import torch.nn as nn
from typing import final

from slora.common.infer_utils import init_bloc
from slora.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from slora.utils.infer_utils import mark_cost_time
from slora.utils.infer_utils import calculate_time, mark_start, mark_end


class LoraPEFTBatchInfer:

    def __init__(self, base_model, infer_adapter=None):
        self.base_model = base_model
        
        self.max_lora_dim = a_len = int(infer_adapter.a_len[0]) // 4
        emb_dim = self.base_model.layers_infer[0].embed_dim_ 

        if infer_adapter is not None:
            self.infer_adapter = infer_adapter
            self.key_buffer = infer_adapter.mem_manager.key_buffer
            self.value_buffer = infer_adapter.mem_manager.value_buffer
            self.adapter_idx = 0

            self.batch_lora_A = [torch.zeros((4, emb_dim, self.max_lora_dim), dtype=torch.float16, device="cuda") for _ in range(self.base_model.layers_num)]
            self.batch_lora_B = [torch.zeros((4, self.max_lora_dim, emb_dim), dtype=torch.float16, device="cuda") for _ in range(self.base_model.layers_num)]
            start = int(infer_adapter.a_start[self.adapter_idx])
            a_len = int(infer_adapter.a_len[self.adapter_idx])
            loc = infer_adapter.a_loc[start:start + a_len]
            r = a_len // 4
            self.scaling = infer_adapter.a_scaling[self.adapter_idx]
            key_buffer = infer_adapter.mem_manager.key_buffer
            value_buffer = infer_adapter.mem_manager.value_buffer

            for layer_id in range(self.base_model.layers_num):
                self.batch_lora_A[layer_id][0,:,:r].copy_(
                        key_buffer[layer_id][loc[:r]].reshape(r, emb_dim).transpose(0, 1))
                # if layer_id == 0:
                #     print("q", self.batch_lora_A[layer_id][0,:10,:10])

                self.batch_lora_A[layer_id][1,:,:r].copy_(
                        key_buffer[layer_id][loc[r:r * 2]].reshape(r, emb_dim).transpose(0, 1))
                self.batch_lora_A[layer_id][2,:,:r].copy_(
                        key_buffer[layer_id][loc[r * 2:r * 3]].reshape(r, emb_dim).transpose(0, 1))
                self.batch_lora_A[layer_id][3,:,:r].copy_(
                        key_buffer[layer_id][loc[r * 3:r * 4]].reshape(r, emb_dim).transpose(0, 1))

                self.batch_lora_B[layer_id][0,:r,:].copy_(
                        value_buffer[layer_id][loc[:r]].reshape(emb_dim, r).transpose(0, 1))
                self.batch_lora_B[layer_id][1,:r,:].copy_(
                        value_buffer[layer_id][loc[r:r * 2]].reshape(emb_dim, r).transpose(0, 1))
                self.batch_lora_B[layer_id][2,:r,:].copy_(
                        value_buffer[layer_id][loc[r * 2:r * 3]].reshape(emb_dim, r).transpose(0, 1))
                self.batch_lora_B[layer_id][3,:r,:].copy_(
                        value_buffer[layer_id][loc[r * 3:r * 4]].reshape(emb_dim, r).transpose(0, 1))
    
    @torch.inference_mode()
    def merge_adapter(self):
        base_model = self.base_model
        for layer_id in range(self.base_model.layers_num):
            base_layer_weight = base_model.trans_layers_weight[layer_id]
            base_layer_infer = base_model.layers_infer[layer_id]
            # AxB
            r = self.infer_adapter.a_len[self.adapter_idx] // 4
            a = self.batch_lora_A[layer_id][:4,:,:r]
            b = self.batch_lora_B[layer_id][:4,:r,:] * self.scaling
            ab = torch.bmm(a.view(4, -1, r), b.view(4, r, -1))
            assert ab.shape == (4, base_layer_infer.embed_dim_, base_layer_infer.embed_dim_)
            # W+AB
            base_layer_weight.q_weight_.add_(ab[0])
            base_layer_weight.k_weight_.add_(ab[1])
            base_layer_weight.v_weight_.add_(ab[2])
            base_layer_weight.o_weight_.add_(ab[3])
    
    @torch.inference_mode()
    def unmerge_adapter(self):
        base_model = self.base_model
        for layer_id in range(self.base_model.layers_num):
            base_layer_weight = base_model.trans_layers_weight[layer_id]
            base_layer_infer = base_model.layers_infer[layer_id]
            # AxB
            r = self.infer_adapter.a_len[self.adapter_idx] // 4
            a = self.batch_lora_A[layer_id][:4,:,:r]
            b = self.batch_lora_B[layer_id][:4,:r,:] * self.scaling
            ab = torch.bmm(a.view(4, -1, r), b.view(4, r, -1))
            assert ab.shape == (4, base_layer_infer.embed_dim_, base_layer_infer.embed_dim_)
            # W-AB
            base_layer_weight.q_weight_.sub_(ab[0])
            base_layer_weight.k_weight_.sub_(ab[1])
            base_layer_weight.v_weight_.sub_(ab[2])
            base_layer_weight.o_weight_.sub_(ab[3])