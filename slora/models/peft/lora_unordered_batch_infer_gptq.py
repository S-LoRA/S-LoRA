import numpy as np
import torch
import torch.nn as nn
from typing import final

from slora.models.peft.lora_unordered_batch_infer import LoraUnorderedBatchInfer
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from slora.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from slora._kernels import dispatch_bgmv

"""
This class inherits from `LoraUnorderedBatchInfer` class.
`LoraUnorderedBatchInferGPTQ` only rewrite some methods to support the forward of quantization layer, e.g. using layer.forward() instead of torch.mm
"""
class LoraUnorderedBatchInferGPTQ(LoraUnorderedBatchInfer):
    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False, no_lora_copy=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # modified
        q = base_layer_weight.q_layer(input_embs.view(-1, base_layer_infer.embed_dim_))
         # @TODO: fix me, filter requests querying only base model
        assert(len(q)==len(self.req_bins))

        # q (bs, H)
        if not no_lora_compute:
            # mark_start("get_q")
            delta_qA = self.delta[0]
            dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 0, self.infer_adapter.a_scaling)
            dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None
            # mark_end("get_q")

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # modified
        base_layer_weight.k_layer(
            input_embs.view(-1, base_layer_infer.embed_dim_), cache = cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)
        )

        # k (bs, H)
        if not no_lora_compute:
            # mark_start("get_k")
            delta_kA = self.delta[1]
            dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None
            # mark_end("get_k")

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # modified
        base_layer_weight.v_layer(
            input_embs.view(-1, base_layer_infer.embed_dim_), cache = cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)
        )

        # v (bs, H)
        if not no_lora_compute:
            # mark_start("get_v")
            delta_vA = self.delta[2]
            dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 2, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
            # mark_end("get_v")

        return q        
    
    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # modified
        q = base_layer_weight.q_layer(input_embs.view(-1, base_layer_infer.embed_dim_))
        assert(len(q)==len(self.batch_req_bins))

        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            delta_qA = self.delta[0]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_qA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_qA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         q, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id],
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling)
                dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)
        
        # modified
        base_layer_weight.k_layer(
            input_embs.view(-1, base_layer_infer.embed_dim_), cache = cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)
        )

        # k (S, H)
        if not no_lora_compute:
            delta_kA = self.delta[1]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_kA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_kA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # modified
        base_layer_weight.v_layer(
            input_embs.view(-1, base_layer_infer.embed_dim_), cache = cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)
        )

        # v (S, H)
        if not no_lora_compute:
            delta_vA = self.delta[2]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
            # if 1 ==0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_vA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_vA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_v.view(-1, base_model.tp_v_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
        return q
    
    def _batch_lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        
        # modified
        o = base_layer_weight.o_layer(input.view(-1, base_layer_infer.embed_dim_))
        if not no_lora_compute:
            # mark_start("get_o")
            delta_oA = self.delta[0]
            dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 3, self.infer_adapter.a_scaling)
            dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 3, self.infer_adapter.a_scaling)
            # delta_oA = None
            # mark_end("get_o")
        return o


    def _lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # modified
        o = base_layer_weight.o_layer(input.view(-1, base_layer_infer.embed_dim_))
        if not no_lora_compute:
            delta_oA = self.delta[0]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_oA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         o, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 3, self.infer_adapter.a_scaling)
            # delta_oA = None
        return o
