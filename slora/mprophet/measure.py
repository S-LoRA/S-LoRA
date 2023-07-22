from slora.mprophet.constants import GB, T, get_num_bytes
from slora.mprophet.hardware_parameters import TFLOPS
from slora.mprophet.model_config import ModelConfig


class ModelProphet:
    name: str
    model_config: ModelConfig


    def __init__(self, name: str, config=None, model_dir=None):
        self.name = name
        self.model_config = ModelConfig(name, config=config, model_dir=model_dir)


    # weights size
    def get_layer_size(self, dtype="fp16"):
        dbytes = get_num_bytes(dtype)
        m = self.model_config
 
        if "opt" in self.name.lower():
            size = dbytes * (
                    # self-attention:
                    m.hidden_size ** 2 * 3 + m.hidden_size ** 2 +
                    # mlp
                    m.hidden_size * m.ffn_embed_dim * 2 +
                    # layer norm
                    m.hidden_size * 4)
            return size
        elif "llama" in self.name.lower():
            size = dbytes * (
                    # self-attention:
                    m.hidden_size ** 2 * 3 + m.hidden_size ** 2 +
                    # mlp
                    m.hidden_size * m.ffn_embed_dim * 3 +
                    # layer norm
                    m.hidden_size * 4)
            return size
        else:
            raise NotImplementedError


    def get_model_size(self, dtype="fp16"):
        return self.get_layer_size(dtype) * self.model_config.num_hidden_layers


    def print_layer_size(self, dtype="fp16"):
        size = self.get_layer_size(dtype)
        print(f"layer size for dtype {dtype}:\n{size / GB:.3f} GB")


    def print_model_size(self, dtype="fp16"):
        size = self.get_model_size(dtype)
        print(f"model size for dtype {dtype}:\n{size / GB:.3f} GB")


    # I/O
    def get_layer_load_time(self, dtype="fp16", bandwidth=1 * GB):
        size = self.get_layer_size(dtype)
        return size / bandwidth


    def get_full_load_time(self, preload=0, bandwidth=1 * GB):
        layer_t = self.get_layer_load_time(bandwidth=bandwidth)
        full_t = self.model_config.num_hidden_layers * layer_t * (1 - preload)


    def print_layer_load_time(self, dtype="fp16", bandwidth=1 * GB):
        t = self.get_layer_load_time(dtype, bandwidth)
        print(f"layer loading time for dtype {dtype} and bandwidth {bandwidth / GB:.2f} GB/s:\n{t:.3f} s")


    # memory
    def get_peak_working_memory(self, bs, context_len, dtype="fp16", tiling_dim = None):
        # if using tiling for attention
        if tiling_dim is not None:
            attn_block_dim = tiling_dim
        else:
            attn_block_dim = context_len

        dbytes = get_num_bytes(dtype)
        m = self.model_config
        mem = dbytes * bs * max(# attention
                                3 * context_len * m.hidden_size +
                                m.n_head * attn_block_dim ** 2 +
                                context_len * m.hidden_size +
                                context_len * m.hidden_size,
                                # mlp
                                context_len * m.hidden_size * 4
                               )
        return mem


    def get_kv_cache_size(self, bs, context_len, dtype="fp16"):
        dbytes = get_num_bytes(dtype)
        m = self.model_config
        return bs * 2 * context_len * m.hidden_size * dbytes


    # compute
    def get_layer_flops(self, token_id, bs, context_len):
        if "opt" in self.name:
            input_len = context_len if token_id == 0 else 1
            # Q, K, V
            m = self.model_config
            flops = 3 * bs * input_len * m.hidden_size * m.hidden_size * 2
            # attention
            head_dim = m.hidden_size // m.n_head
            flops += bs * m.n_head * input_len * head_dim * context_len * 2
            flops += bs * m.n_head * input_len * context_len * head_dim * 2
            # aggregate
            flops += bs * input_len * m.hidden_size * m.hidden_size * 2
            # mlp
            flops += bs * input_len * m.hidden_size * m.hidden_size * 4 * 2
            flops += bs * input_len * m.hidden_size * 4 * m.hidden_size * 2
        else:
            raise NotImplementedError
        return flops


    def get_layer_inference_time(self, token_id, bs, context_len, tflops=None, gpu=None, dtype="fp16"):
        assert not (tflops is None and gpu is None)
        if tflops is None:
            tflops = TFLOPS[gpu]
        flops = self.get_layer_flops(token_id, bs, context_len)
        return flops / T / tflops


    def get_prefill_time(self, context_len, bs):
        layer_t = self.get_layer_inference_time(0, bs, context_len, gpu="3090")
        return layer_t * self.model_config.num_hidden_layers


    def print_layer_inference_time(self, token_id, bs, context_len, tflops=None, gpu=None, dtype="fp16"):
        t = self.get_layer_inference_time(token_id, bs, context_len, tflops, gpu, dtype)
        print(f"layer inference time for token {token_id} with bs {bs} and context length {context_len}:\n{t:.3f} s")


    # others
    def print_model_stats(self, token_id, bs, context_len, tflops):
        print(f"===== Stats for model {self.name} =====")
        self.print_layer_size()
        self.print_layer_load_time(bandwidth=1 * GB)
        self.print_layer_inference_time(token_id, bs, context_len, tflops)
        print()


if __name__ == "__main__":
    model = ModelProphet("opt-30b")
    model.print_model_stats(0, 32, 512, 70)
    model.print_model_stats(1, 32, 512, 70)
    model.print_model_stats(0, 32, 128, 70)
    model.print_model_stats(1, 32, 128, 70)

    model = ModelProphet("opt-175b")
    model.print_model_stats(0, 16, 512, 70)
