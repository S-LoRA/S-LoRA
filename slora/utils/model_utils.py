from transformers.configuration_utils import PretrainedConfig

from slora.mprophet.model_config import get_config_json


def get_model_config(model_dir, dummy=False):
    if dummy:
        try:
            model_cfg = get_config_json(model_dir)
        except NotImplementedError as e:
            model_cfg, _ = PretrainedConfig.get_config_dict(model_dir)
    else:
        model_cfg, _ = PretrainedConfig.get_config_dict(model_dir)
    return model_cfg

