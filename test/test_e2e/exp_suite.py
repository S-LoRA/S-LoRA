from collections import namedtuple
import itertools

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


breakdown_suite = {
    # "test": BenchmarkConfig(
    #     num_adapters = [1],
    #     alpha = [1],
    #     req_rate = [1],
    #     cv = [1],
    #     duration = [1],
    #     input_range = [[8, 512]],
    #     output_range = [[95, 96]],
    # ),

    "test": BenchmarkConfig(
        num_adapters = [4],
        alpha = [0.8],
        req_rate = [6],
        cv = [1],
        duration = [20],
        input_range = [[8, 512]],
        output_range = [[95, 96]],
    ),
}


def get_all_suites(suite="test"):
    
    exps = [{suite: breakdown_suite["test"]}]

    suites = []
    for exp in exps:
        for workload in exp:
            (num_adapters, alpha, req_rate, cv, duration, input_range, output_range) = exp[workload]

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
    ret = (config["num_adapters"], config["alpha"], config["req_rate"], config["cv"],
           config["duration"], tuple(config["input_range"]), tuple(config["output_range"]))
    return ret, keys

