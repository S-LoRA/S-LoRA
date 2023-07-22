import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os
import json

x_label_margin = {
        "Number of Adapters": 0.155,
        "CV Scale": 0.19,
        "Request Rate": 0.18,
        "Alpha": 0.195,
        "Number of Clusters": 0.23,
        }

def plot_single_setting(device, x_name, y_names, legends, setting, data):
    FONTSIZE = 30
    figname = f"Ablation_study_for_adapter_cluster_cv_size_on_{device.replace(' ', '_')}"

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
        if y_names[j] == "Throughput (token/s)":
            ax[j].set_ylim(1)
        else:
            ax[j].set_ylim(0)

        left_margin = 0.18 + 0.45 * j
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

def plot(device, x_name, y_names, legends, settings,
         data_set):
    FONTSIZE = 30
    figname = f"Ablation_study_for_adapter_cluster_size_on_{device.replace(' ', '_')}"

    fig, ax = plt.subplots(len(y_names), len(settings))
    for i, y_name in enumerate(y_names):
        for j, setting in enumerate(settings):
            curves = []
            if len(y_names) == 1:
                ax_single = ax[j]
            else:
                if len(settings) == 1:
                    ax_single = ax[i]
                else:
                    ax_single = ax[i][j]
            for k, d in enumerate(data[i][j]):
                x, y = d
                curves.append(ax_single.plot(x, y, color=f"C{k}", linewidth=4,
                                            marker=".", markersize=FONTSIZE)[0])

            ax_single.grid(True, linestyle='-', linewidth=0.5,
                          alpha=0.5, color="black")
            
            #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            #ax.minorticks_on()
            #ax.grid(which='minor', linestyle=':', linewidth='', color='black')

            ax_single.set_xlim(0)

            if y_name == "Throughput (token/s)":
                ax_single.set_ylim(1)
            else:
                ax_single.set_ylim(0)
            if i == len(y_names) - 1:
                left_margin = x_label_margin[x_name] + 0.4 * j
                vertical = 0.05
                fig.text(left_margin, vertical, x_name, va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            if i == 0:
                left_margin = 0.23 + 0.4 * j
                vertical = 0.91
                fig.text(left_margin, vertical, settings[j], va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            # ax_single.set_xlabel(x_name, fontsize=FONTSIZE)
            ax_single.tick_params(axis='both', which='major',
                                 labelsize=FONTSIZE, length=2, width=1)
            y_format = StrMethodFormatter("{x:.1f}")
            ax_single.yaxis.set_major_formatter(y_format)
            # ax.tick_params(axis='both', which='minor', length=5, width=1)

        left_margin = 0.045
        vertical = 0.7 - 0.42 * i
        fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
                 fontsize=FONTSIZE)
        fig.subplots_adjust(wspace=0.2)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=len(legends) // min(2, (len(legends) - 1) // 4 + 1), fontsize=FONTSIZE)

    # margin = -0.01
    # if max(y) > 10:
    #     margin = -0.03
    # if max(y) > 100:
    #     margin = -0.04
    # Save figure
    fig.set_size_inches((20, 16))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

DEVICE_TAG = {
    "A10G (24GB)": "a10g",
    "A100 (40GB)": "a100"
}

LEGEND_TAG = {
    "SLoRA": "dm",
}

X_TAG = {
    "Number of Clusters": "batch_num_adapters",
}

Y_TAG = {
    "Throughput (token/s)": "throughput",
    "SLO Attainment" : "avg_attainment"
}

SETTING_TAG = {
    "S2 (Llama-7b)": "S2",
    "S4 (Llama-13b)": "S4",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="A100 (40GB)")
    parser.add_argument("--x-name", type=str, default="Number of Clusters")
    parser.add_argument("--y-names", type=str, action="append",
                        default=["Throughput (token/s)", "SLO Attainment"])
    parser.add_argument("--legends", type=str, action="store",
                        default=["alpha = 0.1", "alpha = 0.3", "alpha = 0.6", "alpha = 1"])
    parser.add_argument("--settings", type=str, action="append",
                        default=["S2 (Llama-7b)", "S4 (Llama-13b)"])
    parser.add_argument("--batch-num-adapters", type=int, action="append", default=[1, 2, 4, 8, 32])
    parser.add_argument("--cv", type=bool, default=False)
    args = parser.parse_args()

    if args.cv:
        args.legends = ["cv = 1", "cv = 2", "cv = 4", "cv = 6", "cv = 8"]
        args.settings = ["S2 (Llama-7b)"]

    data = []

    for i in range(len(args.y_names)):
        data.append([])
        if len(args.settings) == 1:
            for k in range(len(args.legends)):
                # TODO extract (x, y): (List, List)
                x = []
                y = []
                for l in range(len(args.batch_num_adapters)):
                    device_tag = DEVICE_TAG[args.device]
                    setting_tag = SETTING_TAG[args.settings[0]]
                    cluster_size_tag = args.batch_num_adapters[l]
                    if args.cv:
                        res_file = f"../paper/ablation_cluster/ablation_cluster_cv_size_{cluster_size_tag}_{device_tag}_{setting_tag}.jsonl"
                    else:
                        res_file = f"../paper/ablation_cluster/ablation_cluster_size_{cluster_size_tag}_{device_tag}_{setting_tag}.jsonl"
                    x_tag = X_TAG[args.x_name]
                    y_tag = Y_TAG[args.y_names[i]]
                    if os.path.exists(res_file):
                        with open(res_file, "r") as f:
                            lines = f.readlines()
                            res = json.loads(lines[k])
                            x.append(cluster_size_tag)
                            y.append(res["result"][y_tag])
                data[i].append((x, y))
        else:
            for j in range(len(args.settings)):
                data[i].append([])
                for k in range(len(args.legends)):
                    x = []
                    y = []
                    for l in range(len(args.batch_num_adapters)):
                        device_tag = DEVICE_TAG[args.device]
                        setting_tag = SETTING_TAG[args.settings[j]]
                        cluster_size_tag = args.batch_num_adapters[l]
                        if args.cv:
                            res_file = f"../paper/ablation_cluster/ablation_cluster_cv_size_{cluster_size_tag}_{device_tag}_{setting_tag}.jsonl"
                        else:
                            res_file = f"../paper/ablation_cluster/ablation_cluster_size_{cluster_size_tag}_{device_tag}_{setting_tag}.jsonl"
                        x_tag = X_TAG[args.x_name]
                        y_tag = Y_TAG[args.y_names[i]]
                        if os.path.exists(res_file):
                            with open(res_file, "r") as f:
                                lines = f.readlines()
                                res = json.loads(lines[k])
                                x.append(cluster_size_tag)
                                y.append(res["result"][y_tag])

                    data[i][j].append((x, y))
    if args.cv:
        plot_single_setting(args.device, args.x_name, args.y_names, args.legends, args.settings[0],
            data)
    else:
        plot(args.device, args.x_name, args.y_names, args.legends, args.settings,
            data)
