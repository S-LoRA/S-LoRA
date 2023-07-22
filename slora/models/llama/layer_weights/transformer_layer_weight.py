import torch
import math
import numpy as np
from slora.common.basemodel import TransformerLayerWeight


class LlamaTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)


    def load_hf_weights(self, weights, dummy=False):
        if dummy:
            self._load_qkvo_dummy_weights()
            self._load_ffn_dummy_weights()
        else:
            self._load_qkvo_weights(weights)
            self._load_ffn_weights(weights)

    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.q_weight_,
                   self.k_weight_,
                   self.v_weight_,
                   self.o_weight_,
                   self.ffn_norm_weight_,
                   self.up_proj,
                   self.gate_proj,
                   self.down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors


    def _load_qkvo_dummy_weights(self):
        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # input layernorm params
        self.att_norm_weight_ = (torch.rand((n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
        # attention params
        self.q_weight_ = (torch.rand((split_n_embed, n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.k_weight_ = (torch.rand((split_n_embed, n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.v_weight_ = (torch.rand((split_n_embed, n_embed), 
                                    dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        # attention output dense params
        self.o_weight_ = (torch.rand((n_embed, split_n_embed),
                                     dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
 

    def _load_ffn_dummy_weights(self):
        n_embed = self.network_config_["hidden_size"]
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        self.ffn_norm_weight_ = (torch.rand((n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3

        self.up_proj = (torch.rand((split_inter_size, n_embed),
                        dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.gate_proj = (torch.rand((split_inter_size, n_embed),
                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        self.down_proj = (torch.rand((n_embed, split_inter_size),
                          dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3


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
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            self.k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(self.k_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            self.v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(self.v_weight_.transpose(0, 1))
        
        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))
    

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])
    
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            self.up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = self._cuda(self.up_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            self.gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = self._cuda(self.gate_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))
