import gc
import os
from safetensors import safe_open
import torch
from tqdm import tqdm

def load_hf_quant_weights(pre_post_datatype, trans_datatype, weight_dir, pre_post_layer=None, transformer_layer_list=None, dummy=False):
    pre_post_datatype = torch.float16 if pre_post_datatype == "fp16" else torch.float32
    if trans_datatype == "int32":
        trans_datatype = torch.int32
    else:
        raise ValueError(f"The trans_datatype should be int32, but got {trans_datatype}")
    
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == pre_post_datatype, f"type is not right, {pre_post_layer.data_type_ } v.s. {pre_post_datatype}"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == trans_datatype, f"type is not right, {transformer_layer_list[0].data_type_} v.s. {trans_datatype}"
    
    if dummy:
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(None, dummy=dummy)
        if transformer_layer_list is not None:
            model_name = weight_dir.rstrip("/").split("/")[-1]
            for layer in tqdm(transformer_layer_list, desc=f"load model {model_name}"):
                layer.load_hf_weights(None, dummy=dummy)
        return

    use_safetensors = True
    files = os.listdir(weight_dir)
    candidate_files = list(filter(lambda x : x.endswith('.safetensors'), files))
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(filter(lambda x : x.endswith('.bin'), files))
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."

    for file_ in candidate_files:
        if use_safetensors:
            weights = safe_open(os.path.join(weight_dir, file_), 'pt', 'cpu')
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
        else:
            weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weights)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weights)
        del weights
        gc.collect()
    return