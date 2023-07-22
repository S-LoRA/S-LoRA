import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import json
import os

from plot_synthetic import plot


x_label_margin = {
        "Number of Adapters": 0.137,
        "CV Scale": 0.19,
        "Request Rate": 0.22,
        "Alpha": 0.195,
        }

y_label_margin = {
        "Throughput (token/s)": 0.06,
        "First Token Latency (s)": 0.06,
        "SLO Attainment": 0.06,
        }


def plot(figname, x_name, y_names, legends, setting, data):
    FONTSIZE = 30

    fig, ax = plt.subplots(1, len(y_names))
    for j, y_name in enumerate(y_names):
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

        left_margin = x_label_margin[x_name] + 0.45 * j
        vertical = -0.05
        fig.text(left_margin, vertical, x_name, va='center',
                 rotation='horizontal', fontsize=FONTSIZE)
        # left_margin = 0.18 + 0.43 * j
        # vertical = 0.95
        # fig.text(left_margin, vertical, settings[j], va='center',
        #          rotation='horizontal', fontsize=FONTSIZE)

        # ax[j].set_xlabel(x_name, fontsize=FONTSIZE)
        ax[j].tick_params(axis='both', which='major',
                             labelsize=FONTSIZE, length=2, width=1)
        y_format = StrMethodFormatter("{x:.1f}")
        ax[j].yaxis.set_major_formatter(y_format)
        # ax.tick_params(axis='both', which='minor', length=5, width=1)

        left_margin = 0.05 + 0.44 * j
        vertical = 0.5
        fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
                 fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.3)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.1),
               ncol=len(legends), fontsize=FONTSIZE)

    # Save figure
    fig.set_size_inches((20, 6))
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


if __name__ == "__main__":
    x_name = "Request Rate"
    y_names = ["Throughput (token/s)", "SLO Attainment"]
    legends = ["SLoRA", "SLoRA-bmm", "SLoRA-no-unify-mem"]
    setting = "A10G (24GB) S2 (Llama-7b)"
    figname = f"Real_workloads_about_the_{X_TAG[x_name]}"

    data = []
    for j, y_name in enumerate(y_names):
        data.append([])
        for k in range(len(legends)):
            pos = setting.index(")") + 1
            device = setting[:pos]
            device_tag = DEVICE_TAG[device]
            legend_tag = LEGEND_TAG[legends[k]]
            setting_tag = SETTING_TAG[setting[pos + 1:]]
            x_tag = X_TAG[x_name]
            y_tag = Y_TAG[y_name]
            res_file = f"../paper/real/{legend_tag}/real_{x_tag}_{device_tag}_{setting_tag}_{legend_tag}.jsonl"
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
            data[j].append((x, y))
    plot(figname, x_name, y_names, legends, setting, data)

  
