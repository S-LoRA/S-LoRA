from collections import namedtuple
import itertools

BASE_MODEL = {
        "S1": "huggyllama/llama-7b",
        "S2": "huggyllama/llama-7b",
        "S3": "huggyllama/llama-13b",
        "S4": "huggyllama/llama-13b",
        "Real": "huggyllama/llama-7b",
}

LORA_DIR = {
        "S1": ["dummy-lora-7b-rank-8"],
        "S2": ["dummy-lora-7b-rank-64", "dummy-lora-7b-rank-32",
               "dummy-lora-7b-rank-16", "dummy-lora-7b-rank-8"],
        "S3": ["dummy-lora-13b-rank-16"],
        "S4": ["dummy-lora-13b-rank-64",
               "dummy-lora-13b-rank-32", "dummy-lora-13b-rank-16",],
        "Real": ["tloen/alpaca-lora-7b", "MBZUAI/bactrian-x-llama-7b-lora"],
}

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    ["num_adapters",
     "alpha", # power law distribution for lambda_i, which are the mean rate for poisson arrival process
     "req_rate", # total request rate per second
     "cv", # coefficient of variation. When cv == 1, the arrival process is Poisson process.
     "duration", # benchmark serving duration
     "input_range", # input length l.b. and u.b.
     "output_range", # output length l.b. and u.b.
    ]
)


paper_suite = {
    "ablation-no-mem": BenchmarkConfig(
        num_adapters = [1, 10, 25, 50, 100, 200],
        alpha = [1],
        req_rate = [2.5],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "ablation-cluster": BenchmarkConfig(
        num_adapters = [32],
        alpha = [0.1, 0.3, 0.6, 1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]]
    ),
    "ablation-cluster-cv": BenchmarkConfig(
        num_adapters = [32],
        alpha = [1, 2, 4, 6, 8],
        req_rate = [2],
        cv = [8],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]]
    ),
    "a10g-num-adapter": BenchmarkConfig(
        num_adapters = [1, 20, 50, 100, 200],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a10g-alpha": BenchmarkConfig(
        num_adapters = [200],
        alpha = [0.1, 0.3, 0.6, 1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a10g-cv": BenchmarkConfig(
        num_adapters = [200],
        alpha = [1],
        req_rate = [2],
        cv = [1,2,4,6,8],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a10g-req-rate": BenchmarkConfig(
        num_adapters = [200],
        alpha = [1],
        req_rate = [1, 1.5, 2, 2.5],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-40-num-adapter-short": BenchmarkConfig(
        num_adapters = [0, 1, 20, 50, 100, 200],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [15],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-40-num-adapter": BenchmarkConfig(
        num_adapters = [0, 1, 20, 50, 100, 200],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a10g-req-rate-real-short": BenchmarkConfig(
        num_adapters = [200],
        alpha = [1],
        req_rate = [1, 2, 3, 4],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a10g-req-rate-real": BenchmarkConfig(
        num_adapters = [200],
        alpha = [1],
        req_rate = [1, 2, 3, 4],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-S2-num-adapter": BenchmarkConfig(
        num_adapters = [1, 20, 50, 100, 200],
        alpha = [1],
        req_rate = [10],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-S2-num-adapter-bmm": BenchmarkConfig(
        num_adapters = [1, 20, 50, 100, 200],
        alpha = [1],
        req_rate = [10],
        cv = [1],
        duration = [30],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-S4-num-adapter": BenchmarkConfig(
        num_adapters = [0, 1, 100, 200, 400],
        alpha = [1],
        req_rate = [6],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-S4-req-rate": BenchmarkConfig(
        num_adapters = [400],
        alpha = [1],
        req_rate = [1, 2, 4, 6, 8],
        cv = [1],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-S4-cv": BenchmarkConfig(
        num_adapters = [400],
        alpha = [1],
        req_rate = [6],
        cv = [1, 2, 4, 6, 8],
        duration = [60 * 5],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-S2-num-adapter-vllm": BenchmarkConfig(
        num_adapters = [5],
        alpha = [1],
        req_rate = [10],
        cv = [1],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-S3-num-adapter-vllm": BenchmarkConfig(
        num_adapters = [2],
        alpha = [1],
        req_rate = [10],
        cv = [1],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-table": BenchmarkConfig(
        num_adapters = [2, 5, 100, 1000, 2000],
        alpha = [1],
        req_rate = [10],
        cv = [1],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-num-adapter-s12-peft": BenchmarkConfig(
        num_adapters = [5, 100],
        alpha = [1],
        req_rate = [10],
        cv = [1],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-80-num-adapter-s4-peft": BenchmarkConfig(
        num_adapters = [2, 100],
        alpha = [1],
        req_rate = [6],
        cv = [1],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
    "a100-40-num-adapter-mp-peft": BenchmarkConfig(
        num_adapters = [2, 100],
        alpha = [1],
        req_rate = [6],
        cv = [1],
        duration = [60 * 2],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
}


breakdown_suite = {
    "a10g": BenchmarkConfig(
        num_adapters = [1, 20, 50, 100, 200],
        alpha = [0.8],
        req_rate = [2],
        cv = [1],
        duration = [60],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "h100": BenchmarkConfig(
        num_adapters = [1, 20, 50, 100, 200, 500, 1000],
        alpha = [0.8],
        req_rate = [20],
        cv = [1],
        duration = [60 * 1],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
}


debug_suite = {
    "default": BenchmarkConfig(
        num_adapters = [100],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "debug": BenchmarkConfig(
        num_adapters = [20],
        alpha = [1],
        req_rate = [4],
        cv = [1],
        duration = [60],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "no-swap": BenchmarkConfig(
        num_adapters = [1],
        alpha = [1],
        req_rate = [20],
        cv = [1],
        duration = [60 * 1],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "swap": BenchmarkConfig(
        num_adapters = [500],
        alpha = [1],
        req_rate = [20],
        cv = [1],
        duration = [60 * 1],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "a10g-no-swap": BenchmarkConfig(
        num_adapters = [1],
        alpha = [0.8],
        req_rate = [2],
        cv = [1],
        duration = [60],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "a10g": BenchmarkConfig(
        num_adapters = [20],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "h100": BenchmarkConfig(
        num_adapters = [1000],
        alpha = [0.8],
        req_rate = [20],
        cv = [1],
        duration = [60],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),

    "ablation-no-mem": BenchmarkConfig(
        # num_adapters = [1, 10, 25, 50, 100, 200],
        num_adapters = [1],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [30],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
    ),
}


def get_all_suites(mode, debug=False, suite=None, breakdown=False):
    assert not (debug and breakdown)
    assert suite is not None
    if debug:
        exps = [{suite: debug_suite[suite]}]
    elif breakdown:
        exps = [{suite: breakdown_suite[suite]}]
    else:
        exps = [{suite: paper_suite[suite]}]

    suites = []
    for exp in exps:
        for workload in exp:
            (num_adapters, alpha, req_rate, cv, duration,
                    input_range, output_range) = exp[workload]
            if mode == "real":
                # These arguments are not used in real trace
                num_adapters = alpha = cv = [None]

            for combination in itertools.product(
                                   num_adapters, alpha, req_rate, cv, duration,
                                   input_range, output_range):
                suites.append(combination)
    return suites


def to_dict(config):
    ret = {}
    for i, key in enumerate(BenchmarkConfig._fields):
        ret[key] = config[i]
    return ret


def to_tuple(config):
    keys = BenchmarkConfig._fields
    ret = (config["num_adapters"], config["alpha"], config["req_rate"],
           config["cv"], config["duration"], tuple(config["input_range"]), tuple(config["output_range"]))
    return ret, keys

