import json
import os


class ModelConfig:
    name: str
    
    def __init__(self, name: str, config=None, model_dir=None):
        self.name = name

        if model_dir is not None:
            with open(os.path.join(model_dir, "config.json"), "r") as f:
                assert config is None or config == json.load(f)
                config = json.load(f)

        if config is not None:
            self._init_from_dict(config)
            return

        if "opt" in name.lower():
            if "opt-125m" in name.lower():
                self.max_seq_len = 2048
                self.num_hidden_layers = 12
                self.n_head=12
                self.hidden_size=768
                self.ffn_embed_dim=768 * 4
            elif "opt-6.7b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=4096 * 4
            elif "opt-13b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=40
                self.n_head=40
                self.hidden_size=5120
                self.ffn_embed_dim=5120 * 4
            elif "opt-30b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=48
                self.n_head=56
                self.hidden_size=7168
                self.ffn_embed_dim=7168 * 4
            elif "opt-175b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=96
                self.n_head=96
                self.hidden_size=12288
                self.ffn_embed_dim=12288 * 4

        elif "llama" in name.lower():
            if "llama-7b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            elif "llama-13b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=40
                self.n_head=40
                self.hidden_size=5120
                self.ffn_embed_dim=13824
            elif "llama-30b-m" in name.lower():
                # Parameters are modified to fit the requirements of custom kernels.
                # Not the official parameters.
                self.max_seq_len=2048
                self.num_hidden_layers=48
                self.n_head=56
                self.hidden_size=7168
                self.ffn_embed_dim=19456
            elif "llama-70b-m" in name.lower():
                # Parameters are modified to fit the requirements of custom kernels.
                # Not the official parameters.
                self.max_seq_len=2048
                self.num_hidden_layers=80
                self.n_head=64
                self.hidden_size=8192
                self.ffn_embed_dim=24576
            elif "llama-14-layer" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=14
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            elif "llama-16-layer" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=16
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            elif "llama-2-7b" in name.lower():
                self.max_seq_len=2048
                self.num_hidden_layers=32
                self.n_head=32
                self.hidden_size=4096
                self.ffn_embed_dim=11008
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


    def _init_from_dict(self, config):
        if "llama" in self.name.lower():
            if "max_sequence_length" in config:
                self.max_seq_len = config["max_sequence_length"]
            else:
                self.max_seq_len = config["max_position_embeddings"]
            self.num_hidden_layers = config["num_hidden_layers"]
            self.n_head = config["num_attention_heads"]
            self.hidden_size = config["hidden_size"]
            self.ffn_embed_dim = config["intermediate_size"]
            self.vocab_size = config["vocab_size"]
        else:
            raise NotImplementedError


def get_config_json(name):
    if "llama-7b" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-13b" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 13824, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 40, "num_hidden_layers": 40, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-30b-m" in name.lower():
        # Parameters are modified to fit the requirements of custom kernels.
        # Not the official parameters.

        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 7168, "intermediate_size": 19456, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 56, "num_hidden_layers": 48, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-70b-m" in name.lower():
        # Parameters are modified to fit the requirements of custom kernels.
        # Not the official parameters.
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 8192, "intermediate_size": 24576, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 64, "num_hidden_layers": 80, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}
    elif "llama-2-7b" in name.lower():
        config = {
            "_name_or_path": "huggyllama/llama-7b",
            "architectures": [
              "LlamaForCausalLM"
            ],
            "attention_bias": False,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "max_sequence_length": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.34.0",
            "use_cache": True,
            "vocab_size": 32000
        }

    elif "llama-14-layer" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 14, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}

    elif "llama-16-layer" in name.lower():
        config = {"architectures": ["LLaMAForCausalLM"], "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 11008, "initializer_range": 0.02, "max_sequence_length": 2048, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 16, "pad_token_id": -1, "rms_norm_eps": 1e-06, "torch_dtype": "float16", "transformers_version": "4.27.0.dev0", "use_cache": True, "vocab_size": 32000}

    else:
        raise NotImplementedError

    return  config
