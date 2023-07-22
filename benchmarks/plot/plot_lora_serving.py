import argparse
import json
import os
import sys
import numpy as np
from plot_utils import plot_two_y, plot_two_y_multi, plot
sys.path.append("..")
from exp_suite import to_tuple


GB = 1024 ** 3


def plot_for_one_dim(data, fix_point, dim, dim_name, dim_set, figname):
    names = ["throughput", "GPU peak memory"]
    x = sorted(dim_set)
    y1 = []
    y2 = []
    for value in x:
        key = list(to_tuple(fix_point)[0])
        key[dim] = value
        key = tuple(key)
        if key not in data:
            break
        y1.append(data[key]["throughput"])
        y2.append(data[key]["gpu_peak_mem"] / GB)
    plot_two_y(names, x[:len(y1)], y1, y2,
               dim_name, "throughput (req/s)", "peak memory (GB)",
               figname)


def plot_for_one_dim_multi(data, fix_points, dim, dim_name, dim_set, figname):
    names_list = []
    y_list = []
    x = sorted(dim_set)
    for fix_point in fix_points:
        names_list.append([f"throughput (alpha={fix_point['alpha']}, req_rate={fix_point['req_rate']})",
                           f"GPU peak memory (alpha={fix_point['alpha']}, req_rate={fix_point['req_rate']})"])
        y1 = []
        y2 = []
        for value in x:
            key = list(to_tuple(fix_point)[0])
            key[dim] = value
            key = tuple(key)
            if key not in data:
                break
            y1.append(data[key]["throughput"])
            y2.append(data[key]["gpu_peak_mem"] / GB)
        y_list.append([y1, y2])

    plot_two_y_multi(names_list, x, y_list,
                     dim_name, "throughput (req/s)", "peak memory (GB)",
                     figname)


def read_data_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        raw_data = [json.loads(line) for line in lines]
    data = {}
    for exp in raw_data:
        config = exp["config"]
        result = exp["result"]
        data[to_tuple(config)[0]] = result
    return data


def get_legend_name(filename):
    if "all_results_dm" in filename and "peft" not in filename:
        return [f"SLoRA",
                f"SLoRA",
                f"SLoRA",
                f"SLoRA",
                f"SLoRA"]
    elif "peft_all_results_dm" in filename:
        return [f"PEFT",
                f"PEFT",
                f"PEFT",
                f"PEFT",
                f"PEFT"]
    elif "no_lora_compute_swap_copy" in filename:
        return [f"throughput (no-lora-compute-swap-copy)",
                f"GPU peak memory (no-lora-compute-swap-copy)"]
    elif "no_lora_compute_swap" in filename:
        return [f"throughput (no-lora-compute-swap)",
                f"GPU peak memory (no-lora-compute-swap)"]
    elif "no_lora_compute" in filename:
        return [f"throughput (no-lora-compute)",
                f"GPU peak memory (no-lora-compute)"]
    elif "no_lora_swap" in filename:
        return [f"throughput (no-lora-swap)",
                f"GPU peak memory (no-lora-swap)"]
    elif "all_results_lora" in filename:
        return [f"throughput (priority queue)",
                f"GPU peak memory (priority queue)"]
    elif "all_results_none" in filename:
        return [f"throughput (original)",
                f"GPU peak memory (original)"]
    name = filename.rstrip("/").split("/")[-1]
    return [name, name]


def plot_breakdown(res_files, dim, dim_name, dim_set, figname, fix_point=None):
    # read data
    names_list = []
    data_list = []
    inter_configs = None
    for filename in res_files:
        names_list.append(get_legend_name(filename))
        data_list.append(read_data_from_file(filename))
        #print(filename, data_list)
        if inter_configs is None:
            inter_configs = set(data_list[-1].keys())
        else:
            inter_configs = inter_configs.intersection(data_list[-1].keys())
    if fix_point is None:
        assert len(inter_configs) > 0
        fix_point = inter_configs.pop()
    print(names_list)
    # prepare plot args
    x = []
    key = list(fix_point)
    for value in sorted(dim_set):
        flag = True
        for data in data_list:
            key[dim] = value
            if tuple(key) not in data:
                flag = False
                break
        if flag:
            x.append(value)
    y_list = []
    # print(data)
    for data in data_list:
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        y5 = []
        for value in x:
            key = list(fix_point)
            key[dim] = value
            key = tuple(key)
            y1.append(data[key]["throughput"])
            y2.append(data[key]["gpu_peak_mem"] / GB)
            y3.append(data[key]["avg_latency"])
            y4.append(data[key]["avg_first_token_latency"])
            y5.append(data[key]["avg_attainment"])
        y_list.append([y1, y2, y3, y4, y5])

    # plot_two_y_multi(names_list, x, y_list,
    #                  dim_name, "throughput (req/s)", "peak memory (GB)",
    #                  figname)
    data1 = [(x, y[0]) for y in y_list]
    plot([x[0] for x in names_list], data1,
         dim_name, "throughput (req/s)", figname + "_throughput")
    data2 = [(x, y[1]) for y in y_list]
    plot([x[1] for x in names_list], data2,
         dim_name, "GPU peak memory (GB)", figname + "_peak_memory")
    data3 = [(x, y[2]) for y in y_list]
    plot([x[2] for x in names_list], data3,
         dim_name, "latency (s)", figname + "_latency")
    data4 = [(x, np.round(y[3],1)) for y in y_list]
    print(data4)
    plot([x[3] for x in names_list], data4,
         dim_name, "first token latency (s)", figname + "_first_token_latency")
    data5 = [(x, y[4]) for y in y_list]
    plot([x[4] for x in names_list], data5,
         dim_name, "attainment (%)", figname + "_attainment")


def get_settings(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        raw_data = [json.loads(line) for line in lines]
    max_throughput = 0
    fix_point = None
    num_adapters_set = set()
    alpha_set = set()
    req_rate_set = set()
    data = {}
    for exp in raw_data:
        config = exp["config"]
        result = exp["result"]
        data[to_tuple(config)[0]] = result

        if result["throughput"] > max_throughput:
            max_throughput = result["throughput"]
            fix_point = config
        num_adapters_set.add(config["num_adapters"])
        alpha_set.add(config["alpha"])
        req_rate_set.add(config["req_rate"])
    return data, fix_point, num_adapters_set, alpha_set, req_rate_set


def plot_ablation(filename):
    data, fix_point, num_adapters_set, alpha_set, req_rate_set = get_settings(filename)
    fix_points = []
    for alpha in alpha_set:
        for req_rate in req_rate_set:
            cur_fix_point = fix_point.copy()
            cur_fix_point["alpha"] = alpha
            cur_fix_point["req_rate"] = req_rate
            fix_points.append(cur_fix_point)

    assert to_tuple(fix_point)[1][0] == "num_adapters"
    plot_for_one_dim_multi(data, fix_points, 0, "number of adapters", list(num_adapters_set),
                           f"lora_serving_plot_num_adapters")
    assert to_tuple(fix_point)[1][1] == "alpha"
    plot_for_one_dim(data, fix_point, 1, "alpha", list(alpha_set),
                     "lora_serving_plot_alpha")
    assert to_tuple(fix_point)[1][2] == "req_rate"
    plot_for_one_dim(data, fix_point, 2, "request rate", list(req_rate_set),
                     "lora_serving_plot_req_rate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../exp_lora/all_results_dm.jsonl")
    parser.add_argument("--breakdown-input", type=str, action="append",
                        default=["../debug_all_results_dm.jsonl",
                                 "../debug_peft_all_results_dm.jsonl",
                                 # "../debug_no_lora_compute_results.jsonl",
                                 # "../debug_no_lora_compute_swap_results.jsonl",
                                 # "../debug_no_lora_swap_results.jsonl",
                                 # "../debug_no_lora_compute_swap_copy_results.jsonl",
                                 # "../debug_all_results_lora.jsonl",
                                 # "../debug_all_results_none.jsonl"
                                 ])
    args = parser.parse_args()

    if os.path.exists(args.input):
        plot_ablation(args.input)
    _, _, num_adapters_set, alpha_set, req_rate_set = get_settings(args.breakdown_input[0])
    plot_breakdown(args.breakdown_input, 0, "number_of_adapters", list(num_adapters_set),
                   "lora_serving_plot_breakdown", fix_point=None)
