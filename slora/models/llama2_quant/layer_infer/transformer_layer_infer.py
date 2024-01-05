import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton

from slora.models.llama2.layer_infer.transformer_layer_infer import Llama2TransformerLayerInfer
from slora.models.llama2_quant.layer_weights.transformer_layer_weight import QuantLlama2TransformerLayerWeight

class QuantLlama2TransformerLayerInfer(Llama2TransformerLayerInfer):
    def _ffn(self, input, infer_state, layer_weight:QuantLlama2TransformerLayerWeight)->torch.Tensor:
        gate_out = layer_weight.gate_layer(input.view(-1, self.embed_dim_))
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = layer_weight.up_layer(input.view(-1, self.embed_dim_))
        input = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = layer_weight.down_layer(ffn1_out)
        ffn1_out = None
        return ffn2_out