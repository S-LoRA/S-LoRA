import os
import json
import torch

from slora.models.llama2_quant.layer_weights.hf_load_utils import load_hf_quant_weights
from slora.models.llama2_quant.layer_infer.transformer_layer_infer import QuantLlama2TransformerLayerInfer
from slora.models.llama2_quant.layer_weights.transformer_layer_weight import QuantLlama2TransformerLayerWeight

from slora.mprophet.model_config import get_quantize_config_json
from slora.utils.model_load import hf_load_quantize_config

from slora.models.llama2.model import Llama2TpPartModel


class QuantLlama2TpPartModel(Llama2TpPartModel):
    # weight class
    transformer_weight_class = QuantLlama2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = QuantLlama2TransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir,
                 max_total_token_num, mem_adapter_size, load_way="HF", mode=[],
                 dummy=False):
        super().__init__(tp_rank, world_size, weight_dir,
                         max_total_token_num, mem_adapter_size, load_way, mode, dummy=dummy)
    

    def _init_weights(self):
        # PS: The weights in `pre_and_post_weight_class` should always have dtype == torch.float16.
        self.pre_post_weight = self.pre_and_post_weight_class(self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.tp_rank_, self.world_size_, torch.int32, network_config=self.config, mode=self.mode)
            for i in range(self.config["n_layer"])
        ]

        # TODO: 实现这个函数, 用来为 transformer layer load int32 的 weights
        load_hf_quant_weights(
            pre_post_datatype = "fp16",
            trans_datatype = "int32",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            dummy=self.dummy
        )

        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]

        # build quant Linear
        [weight.build_quant_layer() for weight in self.trans_layers_weight]

        # exllama buffer
        try:
            from slora.models.gptq.exllama import create_exllama_buffers, set_device
            print("Calling exllama.set_device and exllama.create_exllama_buffers...")
            # set_device(torch.device(f"cuda:{self.tp_rank_}"))
            set_device(torch.device(f"cuda:{self.tp_rank_}"))
            create_exllama_buffers()
        except ImportError:
            print("Error occurs during exllama.set_device and exllama.create_exllama_buffers!")

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        
        # load quantize config
        self._init_quantize_config()
        return 

    def _init_quantize_config(self):
        if self.dummy:
            self.quantize_config = get_quantize_config_json(self.weight_dir_, self.mode)
        else:
            if self.config.get("quantization_config", None) is not None:
                self.quantize_config = self.config["quantization_config"]
            else:
                self.quantize_config = hf_load_quantize_config(self.weight_dir_)
        
        # rename keys

        # bind the quantize_config into config TODO: We will delete this line if it's useless
        self.config["quantize_config"] = self.quantize_config
        
        # register some quantize params
        # TODO: maybe some quantize methods do not have `bits` or `group_size` in the quantize_config
        self.quantize_bits = self.quantize_config["bits"]
        self.quantize_groupsize = self.quantize_config["group_size"]
        
        # TODO: For simplicity, we assert that we always use 4-bit gptq now. We should support other bits later.
        assert self.quantize_bits == 4, f"We should only use the 4-bit gptq now, but got `quantize_bits` = {self.quantize_bits}"

        return
    
    def _verify_params(self):
        super()._verify_params()
        return
