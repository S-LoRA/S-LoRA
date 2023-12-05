import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# from plot.plot_utils import plot
from visualize import (get_req_rate_over_time, get_throughput_over_time, get_service_over_time,
                       get_response_time_over_time, to_client_name,
                       FONTSIZE, MARKERSIZE, legend_x, legend_y, ylabel_x, ylabel_y)


def plot(names, x, ys, x_label, y_label, figname, baseline):
    FONTSIZE = 20
    MARKERSIZE = 4
    legend_x = 0.5
    legend_y = 1.1
    ylabel_x = -0.1
    ylabel_y = 0.5
    markers = ['o','s','v','+','s','D', 'P','X']

    legends = []
    curves = []
    fig, ax = plt.subplots()
    for i, (name, y) in enumerate(zip(names, ys)):
        curves.append(ax.plot(x, y, color=f"C{i}", marker=markers[i], markersize=MARKERSIZE)[0])
        legends.append(to_client_name(name))

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=18)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=21)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"../{baseline}/{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    # baselines = ["VTC", "LCF", "FCFS"]
    baselines = ["LCF"]
    workloads = ["poisson_short_long_2", "increase", "overload", "proportional", "poisson_short_long", "on_off_overload", "on_off_less"]

    for baseline in baselines:
        for workload in workloads:
            exps = []
            input = f"../{baseline}/all_results_{workload}.jsonl"
            with open(input, "r") as f:
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
                T = max([response["req_time"]+response['request_latency'] for response in responses])
                T = int(T) / 10 * 10
                num_x = 100
                window = 60
                x_ticks = [T / num_x * i for i in range(num_x)]

                users = sorted(list(set([response["adapter_dir"] for response in responses])))

                req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users)
                throughput = get_throughput_over_time(responses, T, window, x_ticks, users)
                service = get_service_over_time(responses, T, window, x_ticks, users)
                response_time = get_response_time_over_time(responses, T, window, x_ticks, users)

            # plot
            plot(users, x_ticks, req_rate, "Time (s)", "Request Rate (token/s)", f"sec6.2_{workload}_req_rate", baseline)
            plot(users, x_ticks, throughput, "Time (s)", "Throughput (token/s)", f"sec6.2_{workload}_throughput", baseline)
            plot(users, x_ticks, service, "Time (s)", "Service", f"sec6.2_{workload}_service", baseline)
            plot(users, x_ticks, response_time, "Time (s)", "Response Time (s)", f"sec6.2_{workload}_response_time", baseline)
