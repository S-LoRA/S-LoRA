import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


x_label_margin = {
        "Number of Adapters": 0.155,
        "CV Scale": 0.19,
        "Request Rate": 0.18,
        "Alpha": 0.195,
}


def plot(device, x_name, y_names, legends, settings,
         data_set):
    FONTSIZE = 30
    figname = f"Ablation_study_for_unified_memory_pool_on_{device.replace(' ', '_')}"

    fig, ax = plt.subplots(len(y_names), len(settings))
    for i, y_name in enumerate(y_names):
        for j, setting in enumerate(settings):
            curves = []
            if len(y_names) == 1:
                ax_single = ax[j]
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
            ax_single.set_ylim(0)
            if i == len(y_names) - 1:
                left_margin = x_label_margin[x_name] + 0.2 * j
                vertical = 0.0
                fig.text(left_margin, vertical, x_name, va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            if i == 0:
                left_margin = 0.17 + 0.2 * j
                vertical = 0.95
                fig.text(left_margin, vertical, settings[j], va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            # ax_single.set_xlabel(x_name, fontsize=FONTSIZE)
            ax_single.tick_params(axis='both', which='major',
                                 labelsize=FONTSIZE, length=2, width=1)
            y_format = StrMethodFormatter("{x:.1f}")
            ax_single.yaxis.set_major_formatter(y_format)
            # ax.tick_params(axis='both', which='minor', length=5, width=1)

        left_margin = 0.085
        vertical = 0.5
        fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
                 fontsize=FONTSIZE)
        fig.subplots_adjust(wspace=0.2)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.15),
               ncol=len(legends) // min(2, (len(legends) - 1) // 4 + 1), fontsize=FONTSIZE)

    # margin = -0.01
    # if max(y) > 10:
    #     margin = -0.03
    # if max(y) > 100:
    #     margin = -0.04
    # Save figure
    fig.set_size_inches((40, 8))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


DEVICE_TAG = {
    "A10G (24GB)": "a10g",
}

LEGEND_TAG = {
    "SLoRA": "dm",
    "SLoRA-no-unify-mem": "no_mem",
}

X_TAG = {
    "Number of Adapters": "num_adapters",
}

Y_TAG = {
    "Throughput (token/s)": "throughput",
}

SETTING_TAG = {
    "S1 (Llama-7b)": "S1",
    "S2 (Llama-7b)": "S2",
    "S3 (Llama-13b)": "S3",
    "S4 (Llama-13b)": "S4",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="A10G (24GB)")
    parser.add_argument("--x-name", type=str, default="Number of Adapters")
    parser.add_argument("--y-names", type=str, action="append",
                        default=["Throughput (token/s)"])
    parser.add_argument("--legends", type=str, action="append",
                        default=["SLoRA", "SLoRA-no-unify-mem"])
    parser.add_argument("--settings", type=str, action="append",
                        default=["S1 (Llama-7b)", "S2 (Llama-7b)",
                                 "S3 (Llama-13b)", "S4 (Llama-13b)"])
    args = parser.parse_args()


    data = []
    for i in range(len(args.y_names)):
        data.append([])
        for j in range(len(args.settings)):
            data[i].append([])
            for k in range(len(args.legends)):
                device_tag = DEVICE_TAG[args.device]
                legend_tag = LEGEND_TAG[args.legends[k]]
                setting_tag = SETTING_TAG[args.settings[j]]
                res_file = f"../paper/ablation_mem/ablation_mem_{device_tag}_{setting_tag}_{legend_tag}.jsonl"
                x_tag = X_TAG[args.x_name]
                y_tag = Y_TAG[args.y_names[i]]
                x = []
                y = []
                if os.path.exists(res_file):
                    with open(res_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            res = json.loads(line)
                            x.append(res["config"][x_tag])
                            y.append(res["result"][y_tag])
                print(x, y)
                data[i][j].append((x, y))
    plot(args.device, args.x_name, args.y_names, args.legends, args.settings,
         data)
    
