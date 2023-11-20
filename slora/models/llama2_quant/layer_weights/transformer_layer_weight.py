import torch
import math
import warnings
import numpy as np
import os

MAX_INT = 2147483647
MIN_INT = -2147483648

try:
    major, _minor = torch.cuda.get_device_capability()
except Exception:
    major = 1

USE_EXLLAMA = os.environ.get("USE_EXLLAMA", "True")
USE_EXLLAMA = True if USE_EXLLAMA == "True" else False
HAS_EXLLAMA = False
CAN_EXLLAMA = major >= 8
if CAN_EXLLAMA:
    try:
        from slora.models.gptq.exllama import Ex4bitLinear
        HAS_EXLLAMA = True
    except ImportError:
        warnings.warn(
            f"""
            Can not import `Ex4bitLinear` from exllama, the current compute capability is {major}.{_minor}. `exllama` need compute capability greater than 8.0.

            If your compute capability is greater than 8.0, please check if you have installed the `exllama` package. 

            You can install it follow the following instructions:
            ```
                cd slora/models/exllama
                pip install -v -e .
            ```
            """
        )
        
        # raise NotImplemented("`QuantLlama2TransformerLayerWeight` only support 4-bit gptq quantization now, and use `exllama` to implement the Linear layer. Please make sure you can use the `exllama` and you have installed the `exllama` package.")

from slora.models.gptq.quant_linear import QuantLinear
from slora.common.basemodel import TransformerLayerWeight

from slora.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

"""
As we use the implementation of tgi instead of the implementaton of triton_kernel to build the transformer layer infer process, we will build some layers here.
TODO: In order to keep the consistency with other models' implementations defined in LightLLM, we may need to remove the layers defined in this class and use triton_kernel in the inference process of the transformer layer.

For each FC weights, gptq has:
    1. qweight: torch.int32
    2. qzeros: torch.int32
    3. scales: torch.float16
    4. g_idx: torch.int32
    5. bias: torch.float16
PS: the dtype is the default dtype

For Llama2 model, we use all the former 4 items, but ignore the `bias` following the TGI framework. After checking the existing model weights, all the `bias` are zero tensors.

TODO: Notes: Qwen-14B-Chat-Int4, which uses the auto-gptq, has non-zero tensors.
"""

class QuantLlama2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        # Nowadays, we only support the gptq.
        assert "gptq" in mode, "Please check the `mode` arg, QuantLlama2TransformerLayerWeight only support the gptq quantization now."
        # For simplicity, we assumpt that the world_size is 1 now
        assert world_size == 1, f"For simplicity, `QuantLlama2TransformerLayerWeight` assumpt that the world_size is 1 now, but got {world_size}"
        self.quantize_mode = "gptq"
        self.quantize_config = network_config["quantize_config"]
        self.quantize_bit = self.quantize_config["bits"]
        self.quantize_groupsize = self.quantize_config["group_size"]
        if self.data_type_ == torch.int32:
            self.element_size = 32
        else:
            raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support int32 dtype now.")
    
    def load_hf_weights(self, weights, dummy=False):
        if dummy:
            self._load_qkvo_dummy_weights()
            self._load_ffn_dummy_weights()
        else:
            self._load_qkvo_weights(weights)
            self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"

        fc_prefixs = ["q", "k", "v", "o", "up_proj", "gate_proj", "down_proj"]
        quant_suffixs = []
        if self.quantize_mode == "gptq":
            # Notes: we ignore the bias, you should add bias if you need it
            quant_suffixs += [
                "qweight_", 
                "qzeros_",
                "scales_",
                "g_idx",
            ]
        else:
            raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support gptq now.")
        
        weights = []
        for pre in fc_prefixs:
            for suf in quant_suffixs:
                weights.append(getattr(self, f"{pre}_{suf}"))

        # add weights of the norm layers
        weights += [self.att_norm_weight_, self.ffn_norm_weight_]

        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors

    """
    Func `build_quant_layer` is used to build Quantize Linear layers.
    This func will be called after calling `verify_load` func.
    """
    def build_quant_layer(self):
        if self.quantize_mode == "gptq":
            # check the g_idx
            torch.testing.assert_close(self.q_g_idx, self.k_g_idx)
            torch.testing.assert_close(self.q_g_idx, self.v_g_idx)
            if self.use_exllama:
                self.q_layer = Ex4bitLinear(
                    self.q_qweight_, self.q_qzeros_, self.q_scales_, self.q_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.k_layer = Ex4bitLinear(
                    self.k_qweight_, self.k_qzeros_, self.k_scales_, self.k_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.v_layer = Ex4bitLinear(
                    self.v_qweight_, self.v_qzeros_, self.v_scales_, self.v_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.o_layer = Ex4bitLinear(
                    self.o_qweight_, self.o_qzeros_, self.o_scales_, self.o_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.up_layer = Ex4bitLinear(
                    self.up_proj_qweight_, self.up_proj_qzeros_, self.up_proj_scales_, self.up_proj_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.gate_layer = Ex4bitLinear(
                    self.gate_proj_qweight_, self.gate_proj_qzeros_, self.gate_proj_scales_, self.gate_proj_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.down_layer = Ex4bitLinear(
                    self.down_proj_qweight_, self.down_proj_qzeros_, self.down_proj_scales_, self.down_proj_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
            else:
                self.q_layer = QuantLinear(
                    self.q_qweight_, self.q_qzeros_, self.q_scales_, self.q_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.k_layer = QuantLinear(
                    self.k_qweight_, self.k_qzeros_, self.k_scales_, self.k_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.v_layer = QuantLinear(
                    self.v_qweight_, self.v_qzeros_, self.v_scales_, self.v_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.o_layer = QuantLinear(
                    self.o_qweight_, self.o_qzeros_, self.o_scales_, self.o_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.up_layer = QuantLinear(
                    self.up_proj_qweight_, self.up_proj_qzeros_, self.up_proj_scales_, self.up_proj_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.gate_layer = QuantLinear(
                    self.gate_proj_qweight_, self.gate_proj_qzeros_, self.gate_proj_scales_, self.gate_proj_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                self.down_layer = QuantLinear(
                    self.down_proj_qweight_, self.down_proj_qzeros_, self.down_proj_scales_, self.down_proj_g_idx, None, self.quantize_bit, self.quantize_groupsize
                )
                # raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support exllama-gptq now.")
        else:
            raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support gptq now.")

    def _load_qkvo_dummy_weights(self):
        global USE_EXLLAMA
        norm_n_embed = self.network_config_["hidden_size"]

        # input layernorm params
        self.att_norm_weight_ = (torch.rand((norm_n_embed), dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3

        # attnention params
        if self.quantize_mode == "gptq":
            bit, groupsize = self.quantize_bit, self.quantize_groupsize

            qkv_input_embed = norm_n_embed * self.quantize_bit // self.element_size
            qkv_output_embed = norm_n_embed // self.world_size_
            qkv_qzeros_input_embed = norm_n_embed // groupsize
            qkv_qzeros_output_embed = norm_n_embed * self.quantize_bit // self.element_size // self.world_size_
            qkv_scales_input_embed = qkv_qzeros_input_embed
            qkv_scales_output_embed = qkv_output_embed
            
            o_input_embed = qkv_input_embed // self.world_size_
            o_output_embed = norm_n_embed
            o_qzeros_input_embed = norm_n_embed // groupsize // self.world_size_
            o_qzeros_output_embed = norm_n_embed * self.quantize_bit // self.element_size
            o_scales_input_embed = o_qzeros_input_embed
            o_scales_output_embed = o_output_embed

            self.use_exllama = True if (USE_EXLLAMA and bit == 4) else False

            assert (norm_n_embed % groupsize) == 0

            g_idx_range = torch.arange(norm_n_embed // groupsize, dtype=torch.int32)
            self.q_g_idx = g_idx_range.repeat_interleave(groupsize).cuda()
            self.k_g_idx = g_idx_range.repeat_interleave(groupsize).cuda()
            self.v_g_idx = g_idx_range.repeat_interleave(groupsize).cuda()
            self.o_g_idx = g_idx_range.repeat_interleave(groupsize).cuda()

            self.q_qweight_ = (torch.randint(MIN_INT, MAX_INT, (qkv_input_embed, qkv_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.q_qzeros_ = (torch.randint(MIN_INT, MAX_INT, (qkv_qzeros_input_embed, qkv_qzeros_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.q_scales_ = (torch.rand((qkv_scales_input_embed, qkv_scales_output_embed), 
                                        dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3
            
            self.k_qweight_ = (torch.randint(MIN_INT, MAX_INT, (qkv_input_embed, qkv_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.k_qzeros_ = (torch.randint(MIN_INT, MAX_INT, (qkv_qzeros_input_embed, qkv_qzeros_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.k_scales_ = (torch.rand((qkv_scales_input_embed, qkv_scales_output_embed), 
                                        dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3
            
            self.v_qweight_ = (torch.randint(MIN_INT, MAX_INT, (qkv_input_embed, qkv_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.v_qzeros_ = (torch.randint(MIN_INT, MAX_INT, (qkv_qzeros_input_embed, qkv_qzeros_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.v_scales_ = (torch.rand((qkv_scales_input_embed, qkv_scales_output_embed), 
                                        dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3
            # attention output dense params
            self.o_qweight_ = (torch.randint(MIN_INT, MAX_INT, (o_input_embed, o_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.o_qzeros_ = (torch.randint(MIN_INT, MAX_INT, (o_qzeros_input_embed, o_qzeros_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.o_scales_ = (torch.rand((o_scales_input_embed, o_scales_output_embed), 
                                        dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3
        else:
            raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support gptq now.")

    def _load_ffn_dummy_weights(self):
        norm_n_embed = self.network_config_["hidden_size"]

        self.ffn_norm_weight_ = (torch.rand((norm_n_embed), dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3

        if self.quantize_mode == "gptq":
            bit, groupsize = self.quantize_bit, self.quantize_groupsize
            
            inter_size = self.network_config_["intermediate_size"]

            up_input_embed = norm_n_embed * self.quantize_bit // self.element_size
            up_output_embed = inter_size // self.world_size_
            up_qzeros_input_embed = norm_n_embed // groupsize
            up_qzeros_output_embed = inter_size * self.quantize_bit // self.element_size // self.world_size_
            up_scales_input_embed = up_qzeros_input_embed
            up_scales_output_embed = up_output_embed

            down_input_embed = up_qzeros_output_embed
            down_output_embed = norm_n_embed
            down_qzeros_input_embed = inter_size // groupsize //self.world_size_
            down_qzeros_output_embed = norm_n_embed * self.quantize_bit // self.element_size
            down_scales_input_embed = down_qzeros_input_embed
            down_scales_output_embed = down_output_embed

            # use_exllama = True if bit == 4 else False

            assert (norm_n_embed % groupsize) == 0

            up_g_idx_range = torch.arange(norm_n_embed // groupsize, dtype=torch.int32)
            down_g_idx_range = torch.arange(inter_size // groupsize, dtype=torch.int32)

            self.up_proj_g_idx = up_g_idx_range.repeat_interleave(groupsize).cuda()
            self.gate_proj_g_idx = up_g_idx_range.repeat_interleave(groupsize).cuda()
            self.down_proj_g_idx = down_g_idx_range.repeat_interleave(groupsize).cuda()

            self.up_proj_qweight_ = (torch.randint(MIN_INT, MAX_INT, (up_input_embed, up_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.up_proj_qzeros_ = (torch.randint(MIN_INT, MAX_INT, (up_qzeros_input_embed, up_qzeros_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.up_proj_scales_ = (torch.rand((up_scales_input_embed, up_scales_output_embed), 
                                        dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3
            
            self.gate_proj_qweight_ = (torch.randint(MIN_INT, MAX_INT, (up_input_embed, up_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.gate_proj_qzeros_ = (torch.randint(MIN_INT, MAX_INT, (up_qzeros_input_embed, up_qzeros_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.gate_proj_scales_ = (torch.rand((up_scales_input_embed, up_scales_output_embed), 
                                        dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3
            
            self.down_proj_qweight_ = (torch.randint(MIN_INT, MAX_INT, (down_input_embed, down_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.down_proj_qzeros_ = (torch.randint(MIN_INT, MAX_INT, (down_qzeros_input_embed, down_qzeros_output_embed), 
                                        dtype=self.data_type_, device="cuda"))
            self.down_proj_scales_ = (torch.rand((down_scales_input_embed, down_scales_output_embed), 
                                        dtype=torch.float16, device="cuda") * 2 - 1) * 1e-3
            
        else:
            raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support gptq now.")

    def _load_qkvo_weights(self, weights):
        global USE_EXLLAMA
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])
        
        if self.quantize_mode == "gptq":
            bit, groupsize = self.quantize_bit, self.quantize_groupsize
            self.use_exllama = True if (USE_EXLLAMA and bit == 4) else False

            # if not self.use_exllama:
            #     raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support exllama-gptq now.")

            norm_n_embed = self.network_config_["hidden_size"]
            qkv_input_embed = norm_n_embed * self.quantize_bit // self.element_size
            qkv_output_embed = norm_n_embed // self.world_size_
            qkv_qzeros_input_embed = norm_n_embed // groupsize
            qkv_qzeros_output_embed = norm_n_embed * self.quantize_bit // self.element_size // self.world_size_
            qkv_scales_input_embed = qkv_qzeros_input_embed
            qkv_scales_output_embed = qkv_output_embed
            
            o_input_embed = qkv_input_embed // self.world_size_
            o_output_embed = norm_n_embed
            o_qzeros_input_embed = norm_n_embed // groupsize // self.world_size_
            o_qzeros_output_embed = norm_n_embed * self.quantize_bit // self.element_size
            o_scales_input_embed = o_qzeros_input_embed
            o_scales_output_embed = o_output_embed

            assert (norm_n_embed % groupsize) == 0

            if f"model.layers.{self.layer_num_}.self_attn.o_proj.g_idx" in weights:
                self.o_g_idx = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.g_idx"].cuda()
                # check the o_g_idx is equal to the excepted_g_idx
                if self.world_size_ > 1:
                    g_idx_range = torch.arange(norm_n_embed // groupsize, dtype=torch.int32)
                    excepted_g_idx = g_idx_range.repeat_interleave(groupsize)
                    equal_flag = (self.o_g_idx == 0).all() or torch.equal(self.o_g_idx.cpu(), excepted_g_idx)
                    assert equal_flag

                    self.use_exllama = self.use_exllama and equal_flag
                
            
            if f"model.layers.{self.layer_num_}.self_attn.q_proj.g_idx" in weights:
                self.q_g_idx = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.g_idx"].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.k_proj.g_idx" in weights:
                self.k_g_idx = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.g_idx"].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.v_proj.g_idx" in weights:
                self.v_g_idx = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.g_idx" ].cuda()


            if f"model.layers.{self.layer_num_}.self_attn.q_proj.qweight" in weights:
                self.q_qweight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.qweight"][:, qkv_output_embed * self.tp_rank_ : qkv_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.q_proj.qzeros" in weights:
                self.q_qzeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.qzeros"][:, qkv_qzeros_output_embed * self.tp_rank_ : qkv_qzeros_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.q_proj.scales" in weights:
                self.q_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.scales"][:, qkv_scales_output_embed * self.tp_rank_ : qkv_scales_output_embed * (self.tp_rank_ + 1)].cuda()
            
            if f"model.layers.{self.layer_num_}.self_attn.k_proj.qweight" in weights:
                self.k_qweight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.qweight"][:, qkv_output_embed * self.tp_rank_ : qkv_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.k_proj.qzeros" in weights:
                self.k_qzeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.qzeros"][:, qkv_qzeros_output_embed * self.tp_rank_ : qkv_qzeros_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.k_proj.scales" in weights:
                self.k_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.scales"][:, qkv_scales_output_embed * self.tp_rank_ : qkv_scales_output_embed * (self.tp_rank_ + 1)].cuda()

            if f"model.layers.{self.layer_num_}.self_attn.v_proj.qweight" in weights:
                self.v_qweight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.qweight"][:, qkv_output_embed * self.tp_rank_ : qkv_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.v_proj.qzeros" in weights:
                self.v_qzeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.qzeros"][:, qkv_qzeros_output_embed * self.tp_rank_ : qkv_qzeros_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.v_proj.scales" in weights:
                self.v_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.scales"][:, qkv_scales_output_embed * self.tp_rank_ : qkv_scales_output_embed * (self.tp_rank_ + 1)].cuda()

            if f"model.layers.{self.layer_num_}.self_attn.o_proj.qweight" in weights:
                self.o_qweight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.qweight"][o_input_embed * self.tp_rank_ : o_input_embed * (self.tp_rank_ + 1), :].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.o_proj.qzeros" in weights:
                if groupsize >= 0 or (not self.use_exllama):
                    self.o_qzeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.qzeros"][o_qzeros_input_embed * self.tp_rank_ : o_qzeros_input_embed * (self.tp_rank_ + 1), :].cuda()
                else:
                    self.o_qzeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.qzeros"].cuda()
            if f"model.layers.{self.layer_num_}.self_attn.o_proj.scales" in weights:
                if groupsize >= 0 or (not self.use_exllama):
                    self.o_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.scales"][o_scales_input_embed * self.tp_rank_ : o_scales_input_embed * (self.tp_rank_ + 1), :].cuda()
                else:
                    self.o_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.scales"].cuda()

        else:
            raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support gptq now.")
        
    def _cuda(self, cpu_tensor):
        return cpu_tensor.contiguous().to(torch.float16).cuda()

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])

        if self.quantize_mode == "gptq":
            # if not self.use_exllama:
            #     raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support exllama-gptq now.")
            bit, groupsize = self.quantize_bit, self.quantize_groupsize

            inter_size = self.network_config_["intermediate_size"]
            norm_n_embed = self.network_config_["hidden_size"]

            up_input_embed = norm_n_embed * self.quantize_bit // self.element_size
            up_output_embed = inter_size // self.world_size_
            up_qzeros_input_embed = norm_n_embed // groupsize
            up_qzeros_output_embed = inter_size * self.quantize_bit // self.element_size // self.world_size_
            up_scales_input_embed = up_qzeros_input_embed
            up_scales_output_embed = up_output_embed

            down_input_embed = up_qzeros_output_embed
            down_output_embed = norm_n_embed
            down_qzeros_input_embed = inter_size // groupsize //self.world_size_
            down_qzeros_output_embed = norm_n_embed * self.quantize_bit // self.element_size
            down_scales_input_embed = down_qzeros_input_embed
            down_scales_output_embed = down_output_embed

            assert (norm_n_embed % groupsize) == 0

            if f"model.layers.{self.layer_num_}.mlp.down_proj.g_idx" in weights:
                self.down_proj_g_idx = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.g_idx"].cuda()
                # check the o_g_idx is equal to the excepted_g_idx
                if self.world_size_ > 1:
                    g_idx_range = torch.arange(inter_size // groupsize, dtype=torch.int32)
                    excepted_g_idx = g_idx_range.repeat_interleave(groupsize)
                    equal_flag = (self.o_g_idx == 0).all() or torch.equal(self.o_g_idx.cpu(), excepted_g_idx)
                    assert equal_flag

                    self.use_exllama = self.use_exllama and equal_flag


            if f"model.layers.{self.layer_num_}.mlp.up_proj.g_idx" in weights:
                self.up_proj_g_idx = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.g_idx"].cuda()
            if f"model.layers.{self.layer_num_}.mlp.gate_proj.g_idx" in weights:
                self.gate_proj_g_idx = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.g_idx"].cuda()
            
            if f"model.layers.{self.layer_num_}.mlp.up_proj.qweight" in weights:
                self.up_proj_qweight_ = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.qweight"][:, up_output_embed * self.tp_rank_ : up_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.mlp.up_proj.qzeros" in weights:
                self.up_proj_qzeros_ = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.qzeros"][:, up_qzeros_output_embed * self.tp_rank_ : up_qzeros_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.mlp.up_proj.scales" in weights:
                self.up_proj_scales_ = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.scales"][:, up_scales_output_embed * self.tp_rank_ : up_scales_output_embed * (self.tp_rank_ + 1)].cuda()
                
            if f"model.layers.{self.layer_num_}.mlp.gate_proj.qweight" in weights:
                self.gate_proj_qweight_ = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.qweight"][:, up_output_embed * self.tp_rank_ : up_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.mlp.gate_proj.qzeros" in weights:
                self.gate_proj_qzeros_ = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.qzeros"][:, up_qzeros_output_embed * self.tp_rank_ : up_qzeros_output_embed * (self.tp_rank_ + 1)].cuda()
            if f"model.layers.{self.layer_num_}.mlp.gate_proj.scales" in weights:
                self.gate_proj_scales_ = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.scales"][:, up_scales_output_embed * self.tp_rank_ : up_scales_output_embed * (self.tp_rank_ + 1)].cuda()

            if f"model.layers.{self.layer_num_}.mlp.down_proj.qweight" in weights:
                self.down_proj_qweight_ = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.qweight"][down_output_embed * self.tp_rank_ : down_output_embed * (self.tp_rank_ + 1), :].cuda()
            if f"model.layers.{self.layer_num_}.mlp.down_proj.qzeros" in weights:
                if groupsize >= 0 or (not self.use_exllama):
                    self.down_proj_qzeros_ = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.qzeros"][down_qzeros_output_embed * self.tp_rank_ : down_qzeros_output_embed * (self.tp_rank_ + 1), :].cuda()
                else:
                    self.down_proj_qzeros_ = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.qzeros"].cuda()
            if f"model.layers.{self.layer_num_}.mlp.down_proj.scales" in weights:
                if groupsize >= 0 or (not self.use_exllama):
                    self.down_proj_scales_ = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.scales"][down_scales_output_embed * self.tp_rank_ : down_scales_output_embed * (self.tp_rank_ + 1), :].cuda()
                else:
                    self.down_proj_scales_ = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.scales"].cuda()

        else:
            raise NotImplementedError("`QuantLlama2TransformerLayerWeight` only support gptq now.")
        
"""
if __name__ == "__main__":
import torch
from slora.models.llama2_quant.layer_weights.transformer_layer_weight import QuantLlama2TransformerLayerWeight
from slora.utils.model_load import hf_load_config, hf_load_quantize_config
from slora.models.llama2_quant.layer_weights.hf_load_utils import load_hf_quant_weights

weights_dir = "/data1/bak/suilin/codes/llm_weights/Llama-2-7B-Chat-GPTQ/"
config, _ = hf_load_config(
    weights_dir, mode="model"
)
quantize_config = hf_load_quantize_config(
    weights_dir
)
config["quantize_config"] = quantize_config

layer = QuantLlama2TransformerLayerWeight(
    0, 0, 1, torch.int32, config, mode=["gptq"]
)

load_hf_quant_weights(
    "fp16", "int32", weights_dir, None, [layer], dummy=False
)

layer.build_quant_layer()
"""