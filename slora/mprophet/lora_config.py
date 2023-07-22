import json
import re
import os


class LoRAConfig:
    name: str
    
    def __init__(self, name: str, config=None, weight_dir=None):
        self.name = name

        if weight_dir is not None:
            with open(os.path.join(weight_dir, "adapter_config.json"), "r") as f:
                assert config is None or config == json.load(f)
                config = json.load(f)

        if config is not None:
            self.config = config
            self._init_from_dict(config)
            return

        if "alpaca-lora-7b" in name:
            self.base_model = None
            self.rank = 16
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        elif "bactrian-x-llama-7b-lora" in name:
            self.base_model = None
            self.rank = 64
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        elif "dummy-lora-7b-rank" in name:
            self.base_model = None
            self.rank = int(re.search(r'rank-(\d+)', name).group(1))
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        elif "dummy-lora-13b-rank" in name:
            self.base_model = None
            self.rank = int(re.search(r'rank-(\d+)', name).group(1))
            self.target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    ]
        else:
            raise NotImplementedError

    
    def _init_from_dict(self, config):
        self.base_model = config["base_model_name_or_path"]
        self.rank = config["r"]
        self.target_modules = config["target_modules"]


def get_lora_config_json(name):
    if "alpaca-lora-7b" in name:
        config = {"base_model_name_or_path": "decapoda-research/llama-7b-hf",
                  "bias": "none",
                  "enable_lora": None,
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.05,
                  "merge_weights": False,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": 16,
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    elif "bactrian-x-llama-7b-lora" in name:
        config = {
                  "base_model_name_or_path": "decapoda-research/llama-7b-hf",
                  "bias": "none",
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "init_lora_weights": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.05,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": 64,
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    elif "dummy-lora-7b-rank-" in name:
        config = {"base_model_name_or_path": "huggyllama/llama-7b",
                  "bias": "none",
                  "enable_lora": None,
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.05,
                  "merge_weights": False,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": int(re.search(r'rank-(\d+)', name).group(1)),
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    elif "dummy-lora-13b-rank-" in name:
        config = {"base_model_name_or_path": "meta-llama/Llama-2-13b-hf",
                  "bias": "none",
                  "enable_lora": None,
                  "fan_in_fan_out": False,
                  "inference_mode": True,
                  "lora_alpha": 16,
                  "lora_dropout": 0.1,
                  "merge_weights": False,
                  "modules_to_save": None,
                  "peft_type": "LORA",
                  "r": int(re.search(r'rank-(\d+)', name).group(1)),
                  "target_modules": [
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "o_proj"
                  ],
                  "task_type": "CAUSAL_LM"
                 }
    else:
        raise Exception(f"unrecognized: {name}")
    return config
