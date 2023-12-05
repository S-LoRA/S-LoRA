import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# from plot.plot_utils import plot
from visualize import (get_req_rate_over_time, get_throughput_over_time, get_service_over_time,
                       get_response_time_over_time, to_client_name,
                       FONTSIZE, MARKERSIZE, legend_x, legend_y, ylabel_x, ylabel_y)

def cost_func(input_len, output_len):
    return input_len + 2*output_len

def get_acc_service_diff_over_time(responses, T, window, x_ticks, users):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = 0
            r = x
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    if end_time < 1:
                        continue
                    service = cost_func(response['prompt_len'],response["output_len"])
                    overlap = max(min(r, end_time) - max(l, start_time), 0)
                    y[-1][i] += service * overlap / (end_time - start_time)
    # compute max difference in service over time
    max_diff = []
    for i in range(len(x_ticks)):
        max_diff.append(max([y[j][i] for j in range(len(users))]) - min([y[j][i] for j in range(len(users))]))
    return max_diff

def plot(baslines, x, ys, x_label, y_label, figname):
    FONTSIZE = 20
    MARKERSIZE = 4
    legend_x = 0.5
    legend_y = 1.1
    ylabel_x = 0
    ylabel_y = 0.5
    markers = ['o','s','v','+','s','D', 'P','X']

    legends = []
    curves = []
    fig, ax = plt.subplots()
    for i, (name, y) in enumerate(zip(baslines, ys)):
        curves.append(ax.plot(x, y, color=f"C{i}", marker=markers[i], markersize=MARKERSIZE)[0])
        legends.append(name)

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")

    ax.set_xlim(1)
    ax.set_ylim(0)
    ax.set_ylim(0,300000)
    ax.set_xlabel(x_label, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    # ax.yaxis.set_major_formatter(y_format)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=18)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=21)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../all_results_proportional.jsonl")
    args = parser.parse_args()

    # baselines = ['VTC','LCF','FCFS']
    workloads = ['poisson_short_long','poisson_short_long_2','overload','on_off_overload']
    
    for workload in workloads:
        acc_services_diffs = []
        if workload == 'poisson_short_long' or workload == 'poisson_short_long_2' or workload == 'overload':
            baselines = ['VTC','FCFS']
        else:
            baselines = ['VTC','FCFS','LCF']
        for baseline in baselines:
            path = f"../{baseline}/all_results_{workload}.jsonl"
            exps = []
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    exps.append({})
                    exps[-1]["config"] = json.loads(line)["config"]
                    exps[-1]["result"] = json.loads(line)["result"]

            # get data points
            for exp in exps:
                config = exp["config"]
                result = exp["result"]

                responses = result["responses"]
                T = max([response["req_time"] for response in responses])
                T = int(T) / 10 * 10
                num_x = 100
                window = 60
                x_ticks = [T / num_x * i for i in range(num_x)]
                x_ticks = x_ticks[1:]

                users = sorted(list(set([response["adapter_dir"] for response in responses])))
                acc_service_diff = get_acc_service_diff_over_time(responses, T, window, x_ticks, users)
                acc_services_diffs.append(acc_service_diff)
                
            

        # plot
        plot(baselines, x_ticks, acc_services_diffs, "Time (s)", "Absolute Difference in Service", f"sec6.2_{workload}_acc_service_diff")
