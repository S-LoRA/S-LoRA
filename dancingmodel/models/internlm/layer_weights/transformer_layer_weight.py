import torch
import math
import numpy as np
from dancingmodel.common.basemodel import TransformerLayerWeight

from dancingmodel.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

class InternlmTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return
    
    def verify_load(self):
        errors = "weights load not ok"
         
        # handle internlm 20b, which has no bias, so set q k v o bias to zero
        if not self.network_config_.get("bias", True):
            for layer_type in ("q", "k", "v", "o"):
                attr_name = f"{layer_type}_bias_"
                if hasattr(self, attr_name):
                    continue
                setattr(self, attr_name, self._cuda(torch.zeros(1)))

        weights = [self.att_norm_weight_,
                   self.q_weight_,
                   self.k_weight_,
                   self.v_weight_,
                   self.o_weight_,
                   self.q_bias_,
                   self.k_bias_,
                   self.v_bias_,
                   self.o_bias_,
                   self.ffn_norm_weight_,
                   self.up_proj,
                   self.gate_proj,
                   self.down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 
    
    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.bias" in weights:
            self.q_bias_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.q_bias_ = self._cuda(self.q_bias_)
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            self.k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(self.k_weight_.transpose(0, 1))
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.bias" in weights:
            self.k_bias_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.k_bias_ = self._cuda(self.k_bias_)
        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            self.v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(self.v_weight_.transpose(0, 1))
        if f"model.layers.{self.layer_num_}.self_attn.v_proj.bias" in weights:
            self.v_bias_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.v_bias_ = self._cuda(self.v_bias_)        
        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.bias" in weights:
            self.o_bias_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"]
            self.o_bias_ = self._cuda(self.o_bias_)   
        return
