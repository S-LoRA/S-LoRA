import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import json
import os

from plot_synthetic import plot


x_label_margin = {
        "Number of Adapters": 0.137,
        "CV Scale": 0.19,
        "Request Rate": 0.24,
        "Alpha": 0.195,
        }

y_label_margin = {
        "Throughput (token/s)": 0.06,
        "First Token Latency (s)": 0.06,
        "SLO Attainment": 0.06,
        }


def plot(figname, x_name, y_names, legends, settings, data):
    FONTSIZE = 30

    fig, ax = plt.subplots(len(y_names), len(settings))
    for i, y_name in enumerate(y_names):
        for j, setting in enumerate(settings):
            curves = []
            min_y = 1000
            max_y = 0
            for k, d in enumerate(data[i][j]):
                x, y = d
                if x == []: continue
                min_y = min(min_y, max(y))
                max_y = max(max_y, max(y))
                curves.append(ax[i][j].plot(x, y, color=f"C{k}", linewidth=4,
                                            marker=".", markersize=FONTSIZE)[0])

            ax[i][j].grid(True, linestyle='-', linewidth=0.5,
                          alpha=0.5, color="black")
            
            #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            #ax.minorticks_on()
            #ax.grid(which='minor', linestyle=':', linewidth='', color='black')

            # ax[i][j].set_xlim(0)
            if "Latency" in y_name:
                ax[i][j].set_ylim(0, min(min_y * 3, max_y * 1.1))
            else:
                ax[i][j].set_ylim(0)
            if i == len(y_names) - 1:
                left_margin = x_label_margin[x_name] + 0.161 * j
                vertical = 0.04
                fig.text(left_margin, vertical, x_name, va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            if i == 0:
                left_margin = 0.155 + 0.16 * j
                vertical = 0.93
                fig.text(left_margin, vertical, settings[j], va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            # ax[i][j].set_xlabel(x_name, fontsize=FONTSIZE)
            ax[i][j].tick_params(axis='both', which='major',
                                 labelsize=FONTSIZE, length=2, width=1)
            y_format = StrMethodFormatter("{x:.1f}")
            ax[i][j].yaxis.set_major_formatter(y_format)
            # ax.tick_params(axis='both', which='minor', length=5, width=1)

        left_margin = 0.085
        vertical = 0.7 - 0.42 * i
        fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
                 fontsize=FONTSIZE)
        fig.subplots_adjust(wspace=0.26)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.07),
               ncol=len(legends), fontsize=FONTSIZE)

    # Save figure
    fig.set_size_inches((40, 13))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


def plot2(figname, x_name, y_names, legends, settings, data):
    FONTSIZE = 30

    fig, ax = plt.subplots(len(y_names), len(settings))
    for i, y_name in enumerate(y_names):
        for j, setting in enumerate(settings):
            curves = []
            min_y = 1000
            max_y = 0
            for k, d in enumerate(data[i][j]):
                x, y = d
                if x == []: continue
                min_y = min(min_y, max(y))
                max_y = max(max_y, max(y))
                curves.append(ax[i][j].plot(x, y, color=f"C{k}", linewidth=4,
                                            marker=".", markersize=FONTSIZE)[0])

            ax[i][j].grid(True, linestyle='-', linewidth=0.5,
                          alpha=0.5, color="black")
            
            #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            #ax.minorticks_on()
            #ax.grid(which='minor', linestyle=':', linewidth='', color='black')

            # ax[i][j].set_xlim(0)
            if "Latency" in y_name:
                ax[i][j].set_ylim(0, min(min_y * 3, max_y * 1.1))
            else:
                ax[i][j].set_ylim(0)
            if i == len(y_names) - 1:
                left_margin = x_label_margin[x_name] + 0.4 * j
                vertical = 0.05
                fig.text(left_margin, vertical, x_name, va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            if i == 0:
                left_margin = 0.155 + 0.42 * j
                vertical = 0.915
                fig.text(left_margin, vertical, settings[j], va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            # ax[i][j].set_xlabel(x_name, fontsize=FONTSIZE)
            ax[i][j].tick_params(axis='both', which='major',
                                 labelsize=FONTSIZE, length=2, width=1)
            y_format = StrMethodFormatter("{x:.1f}")
            ax[i][j].yaxis.set_major_formatter(y_format)
            # ax.tick_params(axis='both', which='minor', length=5, width=1)

        left_margin = 0.04
        if i < 2:
            vertical = 0.77 - 0.277 * i
        else:
            vertical = 0.77 - 0.28 * i
        fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
                 fontsize=FONTSIZE)
        fig.subplots_adjust(wspace=0.2)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1),
               ncol=len(legends), fontsize=FONTSIZE)

    # Save figure
    fig.set_size_inches((20, 18))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


def plot_single_row(figname, x_name, y_name, legends, settings, data):
    FONTSIZE = 25

    fig, ax = plt.subplots(1, len(settings))
    for j, setting in enumerate(settings):
        curves = []
        for k, d in enumerate(data[j]):
            x, y = d
            curves.append(ax[j].plot(x, y, color=f"C{k}", linewidth=3,
                                        marker=".", markersize=FONTSIZE)[0])

        ax[j].grid(True, linestyle='-', linewidth=0.5,
                      alpha=0.5, color="black")
        
        #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        #ax.minorticks_on()
        #ax.grid(which='minor', linestyle=':', linewidth='', color='black')

        # ax[j].set_xlim(0)
        ax[j].set_ylim(0)

        left_margin = x_label_margin[x_name] + 0.455 * j
        vertical = -0.05
        fig.text(left_margin, vertical, x_name, va='center',
                 rotation='horizontal', fontsize=FONTSIZE)
        left_margin = 0.18 + 0.43 * j
        vertical = 0.95
        fig.text(left_margin, vertical, settings[j], va='center',
                 rotation='horizontal', fontsize=FONTSIZE)

        # ax[j].set_xlabel(x_name, fontsize=FONTSIZE)
        ax[j].tick_params(axis='both', which='major',
                             labelsize=FONTSIZE, length=2, width=1)
        y_format = StrMethodFormatter("{x:.1f}")
        ax[j].yaxis.set_major_formatter(y_format)
        # ax.tick_params(axis='both', which='minor', length=5, width=1)

    left_margin = y_label_margin[y_name]
    vertical = 0.5
    fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
             fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.2),
               ncol=len(legends), fontsize=FONTSIZE)

    # Save figure
    fig.set_size_inches((20, 5))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

DEVICE_TAG = {
    "A10G (24GB)": "a10g",
    "A100 (40GB)": "a100-40",
    "A100 (80GB)": "a100-80",
}

LEGEND_TAG = {
    "SLoRA": "slora",
    "SLoRA-Abort": "abort",
    "SLoRA-PetS": "pets",
    "SLoRA-bmm": "bmm",
    "SLoRA-no-unify-mem": "no_mem",
    "PEFT": "peft",
    "vLLM-packed": "vllm",
}

X_TAG = {
    "Number of Adapters": "num_adapters",
    "Request Rate": "req_rate",
    "Alpha": "alpha",
    "CV Scale": "cv",
}

Y_TAG = {
    "Throughput (token/s)": "throughput",
    "Average Latency (s)": "avg_latency",
    "First Token Latency (s)": "avg_first_token_latency",
    "User Satisfaction": "avg_satisfaction",
    "SLO Attainment": "avg_attainment",
}

SETTING_TAG = {
    "S1 (Llama-7b)": "S1",
    "S2 (Llama-7b)": "S2",
    "S3 (Llama-13b)": "S3",
    "S4 (Llama-13b)": "S4",
}


def plot_num_adapter():
    x_name = "Number of Adapters"
    y_names = ["Throughput (token/s)", "Average Latency (s)"]
    legends = ["SLoRA", "SLoRA-bmm", "SLoRA-no-unify-mem"]
    settings = ["A10G (24GB)\nS1 (Llama-7b)", "A10G (24GB)\nS2 (Llama-7b)",
                "A100 (40GB)\nS4 (Llama-13b)",
                "A100 (80GB)\nS2 (Llama-7b)", "A100 (80GB)\nS4 (Llama-13b)"]
    r_settings = ['\n'.join(reversed(s.split('\n'))) for s in settings]
    figname = f"Synthetic_workloads_about_the_{x_name.replace(' ', '_')}"

    data = []
    for i in range(len(y_names)):
        data.append([])
        for j, setting in enumerate(settings):
            data[i].append([])
            for k in range(len(legends)):
                pos = setting.index(")") + 1
                device = setting[:pos]
                device_tag = DEVICE_TAG[device]
                legend_tag = LEGEND_TAG[legends[k]]
                setting_tag = SETTING_TAG[setting[pos + 1:]]
                x_tag = X_TAG[x_name]
                y_tag = Y_TAG[y_names[i]]
                res_file = f"../paper/synthetic/{legend_tag}/synthetic_{x_tag}_{device_tag}_{setting_tag}_{legend_tag}.jsonl"
                x = []
                y = []
                if os.path.exists(res_file):
                    print(f"{res_file} exists")
                    with open(res_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            res = json.loads(line)
                            if res["config"][x_tag] == 0:
                                continue
                            x.append(res["config"][x_tag])
                            y.append(res["result"][y_tag])
                else:
                    print(f"{res_file} not exists")
                    x = []
                    y = []
                print(x, y)
                data[i][j].append((x, y))
    plot(figname, x_name, y_names, legends, r_settings, data)


def plot_req_rate():
    x_name = "Request Rate"
    y_names = ["Throughput (token/s)", "First Token Latency (s)", "SLO Attainment"]
    legends = ["SLoRA", "SLoRA-bmm", "SLoRA-no-unify-mem"]
    settings = ["A10G (24GB) S2 (Llama-7b)", "A100 (80GB) S4 (Llama-13b)"]
    figname = f"Synthetic_workloads_about_the_{X_TAG[x_name]}"
    r_settings = ["S2 (Llama-7b) A10G (24GB)", "S4 (Llama-13b) A100 (80GB)"]

    data = []
    for i, y_name in enumerate(y_names):
        data.append([])
        for j, setting in enumerate(settings):
            data[i].append([])
            for k in range(len(legends)):
                pos = setting.index(")") + 1
                device = setting[:pos]
                device_tag = DEVICE_TAG[device]
                legend_tag = LEGEND_TAG[legends[k]]
                setting_tag = SETTING_TAG[setting[pos + 1:]]
                x_tag = X_TAG[x_name]
                y_tag = Y_TAG[y_name]
                res_file = f"../paper/synthetic/{legend_tag}/synthetic_{x_tag}_{device_tag}_{setting_tag}_{legend_tag}.jsonl"
                x = []
                y = []
                if os.path.exists(res_file):
                    with open(res_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            res = json.loads(line)
                            x.append(res["config"][x_tag])
                            y.append(res["result"][y_tag])
                else:
                    x = []
                    y = []
                print(x, y)
                data[i][j].append((x, y))
    plot2(figname, x_name, y_names, legends, r_settings, data)


if __name__ == "__main__":
    plot_num_adapter()
    plot_req_rate()
   
