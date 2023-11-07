import time
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from dancingmodel.common.basemodel import TransformerLayerInferTpl
from dancingmodel.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from dancingmodel.models.bloom.triton_kernel.context_flashattention_nopad import context_attention_fwd
from dancingmodel.models.bloom.triton_kernel.token_flashattention_nopad import token_attention_fwd
from dancingmodel.models.bloom.triton_kernel.layernorm import layernorm_forward
from dancingmodel.common.basemodel import InferStateInfo
from dancingmodel.utils.infer_utils import mark_cost_time


class BloomTransformerLayerInfer(TransformerLayerInferTpl):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_epsilon"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = self.tp_q_head_num_
        self.tp_v_head_num_ = self.tp_q_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["n_embed"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["n_embed"]
        return
    
    def _att_norm(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.att_norm_weight_,
            bias=layer_weight.att_norm_bias_,
            eps=self.eps_)
    
    def _ffn_norm(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.ffn_norm_weight_,
            bias=layer_weight.ffn_norm_bias_,
            eps=self.eps_)
    
    def _get_qkv(self, input, cache_k, cache_v, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        q = torch.addmm(layer_weight.q_bias_, input.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        torch.addmm(layer_weight.k_bias_, input.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0,
                    alpha=1.0, out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        torch.addmm(layer_weight.v_bias_, input.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0,
                    alpha=1.0, out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q
    
    def _context_attention_kernel(self, q, k, v, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              layer_weight.tp_alibi,
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor
    
    def _token_attention_kernel(self, q, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        token_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                            infer_state.mem_manager.key_buffer[self.layer_num_],
                            infer_state.mem_manager.value_buffer[self.layer_num_],
                            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                            layer_weight.tp_alibi,
                            infer_state.b_loc,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)
        return o_tensor
    
    def _get_o(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o = torch.addmm(layer_weight.o_bias_,
                        input.view(-1, self.tp_q_head_num_ * self.head_dim_),
                        layer_weight.o_weight_,
                        beta=1.0 / self.world_size_)
        return o
    
    def _ffn(self, input, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        ffn1_out = torch.addmm(layer_weight.ffn_1_bias_, input.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate='tanh')
        ffn1_out = None
        ffn2_out = torch.addmm(layer_weight.ffn_2_bias_, gelu_out, layer_weight.ffn_2_weight_, beta=1.0 / self.world_size_)
        gelu_out = None
        return ffn2_out