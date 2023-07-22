import torch
from tqdm import tqdm
import os
import gc
from safetensors import safe_open


def load_hf_weights(data_type, weight_dir, transformer_layer_list=None,
                    swap=False, dummy=False):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"

    if dummy:
        for layer in transformer_layer_list:
            layer.load_hf_weights(None, swap=swap, dummy=dummy)
        return

    use_safetensors = True
    files = os.listdir(weight_dir)
    candidate_files = list(filter(lambda x : x.endswith('.safetensors'), files))
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(filter(lambda x : x.endswith('.bin'), files))
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."

    model_name = weight_dir.rstrip("/").split("/")[-1]
    # for file_ in tqdm(candidate_files, desc=f"load {model_name}"):
    for file_ in candidate_files:
        if use_safetensors:
            weights = safe_open(os.path.join(weight_dir, file_), 'pt', 'cpu')
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
        else:
            weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
        # for key, tensor in weights.items():
        #     if "layers.0" in key:
        #         print(key)
        #         print(tensor.shape)

        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weights, swap=swap)
        del weights
        gc.collect()
    return


# load_hf_weights("fp16", "/home/ubuntu/models/lora-adapters/llama-7b/alpaca-lora-7b")
