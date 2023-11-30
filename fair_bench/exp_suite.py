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
     "on_off",
     "mode",
    ]
)


paper_suite = {
    "ablation_lshare": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 5]],
        cv = [-1],
        duration = [60],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "overload": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[1.5, 3]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "proportional": BenchmarkConfig(
        num_adapters = [3],
        alpha = [-1],
        req_rate = [[0.25, 0.5, 1.5]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["uniform"],
    ),
    "increase": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["increase"],
    ),
    "on_off_less": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [60],
        mode = ["uniform"],
    ),
    "on_off_overload": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[2, 3]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [60],
        mode = ["uniform"],
    ),
    "poisson_on_off_overload": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[16, 512]],
        output_range = [[16, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),
    "poisson_short_long": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[8, 1.5]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[256, 257]],
        output_range = [[256, 257]],
        on_off = [-1],
        mode = ["poisson-short-long"],
    ),
    "poisson_short_long_2": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[8, 1.5]],
        cv = [-1],
        duration = [60 * 10],
        input_range = [[32, 33]],
        output_range = [[512, 513]],
        on_off = [-1],
        mode = ["poisson-short-long-2"],
    ),
}


debug_suite = {
    "default": BenchmarkConfig(
        num_adapters = [2],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 1],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "diff_slo": BenchmarkConfig(
        num_adapters = [2],
        alpha = [1],
        req_rate = [2],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "increase": BenchmarkConfig(
        num_adapters = [2],
        alpha = [-1],
        req_rate = [[0.5, 2]],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [-1],
        mode = ["increase"],
    ),

    "on_off": BenchmarkConfig(
        num_adapters = [2],
        alpha = [0.4],
        req_rate = [3],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "on_off_equal": BenchmarkConfig(
        num_adapters = [2],
        alpha = [1],
        req_rate = [3],
        cv = [1],
        duration = [60 * 10],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [60],
        mode = ["poisson"],
    ),

    "unbalance": BenchmarkConfig(
        num_adapters = [2],
        alpha = [0.2],
        req_rate = [4],
        cv = [1],
        duration = [60 * 20],
        input_range = [[8, 512]],
        output_range = [[8, 512]],
        on_off = [-1],
        mode = ["poisson"],
    ),
}


def get_all_suites(debug=False, suite=None, breakdown=False):
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
                    input_range, output_range, on_off, mode) = exp[workload]
            for combination in itertools.product(
                                   num_adapters, alpha, req_rate, cv, duration,
                                   input_range, output_range, on_off, mode):
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
           config["cv"], config["duration"],
           tuple(config["input_range"]), tuple(config["output_range"]),
           config["on_off"], config["mode"])
    return ret, keys

