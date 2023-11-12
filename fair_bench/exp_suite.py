from collections import namedtuple
import itertools

BASE_MODEL = {
        "S1": "huggyllama/llama-7b",
        "Real": "huggyllama/llama-7b",
}

LORA_DIR = {
        "S1": ["dummy-lora-7b-rank-8"],
        "Real": ["tloen/alpaca-lora-7b"],
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


debug_suite = {
    "default": BenchmarkConfig(
        num_adapters = [2],
        alpha = [0.5],
        req_rate = [4],
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

