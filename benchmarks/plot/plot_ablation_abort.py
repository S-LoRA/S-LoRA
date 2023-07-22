import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import json
import os


x_label_margin = {
        "Number of Adapters": 0.15,
        "CV Scale": 0.27,
        "Request Rate": 0.18,
        "Alpha": 0.195,
        }


def plot(device, x_name, y_names, legends, settings,
         data_set):
    FONTSIZE = 18
    figname = f"Ablation_abort_on_{device.replace(' ', '_')}" \
              f"_about_the_{x_name.replace(' ', '_')}"

    fig, ax = plt.subplots(len(y_names), len(settings))
    for i, y_name in enumerate(y_names):
        for j, setting in enumerate(settings):
            curves = []
            for k, d in enumerate(data[i][j]):
                x, y = d
                curves.append(ax[i][j].plot(x, y, color=f"C{k}", linewidth=3,
                                            marker=".", markersize=FONTSIZE)[0])

            ax[i][j].grid(True, linestyle='-', linewidth=0.5,
                          alpha=0.5, color="black")
            
            #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            #ax.minorticks_on()
            #ax.grid(which='minor', linestyle=':', linewidth='', color='black')

            # ax[i][j].set_xlim(0)
            ax[i][j].set_ylim(0)
            if i == len(y_names) - 1:
                left_margin = x_label_margin[x_name] + 0.42 * j
                vertical = 0.05
                fig.text(left_margin, vertical, x_name, va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            if i == 0:
                left_margin = 0.25 + 0.42 * j
                vertical = 0.90
                fig.text(left_margin, vertical, settings[j], va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            # ax[i][j].set_xlabel(x_name, fontsize=FONTSIZE)
            ax[i][j].tick_params(axis='both', which='major',
                                 labelsize=FONTSIZE, length=2, width=1)
            y_format = StrMethodFormatter("{x:.1f}")
            ax[i][j].yaxis.set_major_formatter(y_format)
            # ax.tick_params(axis='both', which='minor', length=5, width=1)

        left_margin = 0.05
        vertical = 0.72 - 0.45 * i
        fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
                 fontsize=FONTSIZE)
        fig.subplots_adjust(wspace=0.2)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(legends) // min(2, (len(legends) - 1) // 4 + 1), fontsize=FONTSIZE)

    # margin = -0.01
    # if max(y) > 10:
    #     margin = -0.03
    # if max(y) > 100:
    #     margin = -0.04
    # Save figure
    fig.set_size_inches((10, 7))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

DEVICE_TAG = {
    "A10G (24GB)": "a10g",
    "A100 (80GB)": "a100-80",
}

LEGEND_TAG = {
    "SLoRA-FCFS": "slora",
    "SLoRA-Abort": "abort",
    "SLoRA-LCFS": "lifo",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="A10G (24GB)")
    parser.add_argument("--x-name", type=str, default="CV Scale",
                        choices=["Number of Adapters",
                                 "Request Rate",
                                 "Alpha",
                                 "CV Scale",
                                ])
    parser.add_argument("--y-names", type=str, action="append",
                        default=["SLO Attainment", "User Satisfaction"])
    parser.add_argument("--legends", type=str, action="append",
                        default=["SLoRA-FCFS", "SLoRA-LCFS", "SLoRA-Abort"])
    parser.add_argument("--settings", type=str, action="append",
                        default=["S2 (Llama-7b)", "S4 (Llama-13b)"])
    args = parser.parse_args()

    settings = ["A10G (24GB)\nS2 (Llama-7b)", "A100 (80GB)\nS4 (Llama-13b)"]

    data = []
    for i in range(len(args.y_names)):
        data.append([])
        for j, setting in enumerate(settings):
            data[i].append([])
            for k in range(len(args.legends)):
                pos = setting.index(")") + 1
                device = setting[:pos]
                device_tag = DEVICE_TAG[device]
                setting_tag = SETTING_TAG[setting[pos + 1:]]
                legend_tag = LEGEND_TAG[args.legends[k]]
                x_tag = X_TAG[args.x_name]
                y_tag = Y_TAG[args.y_names[i]]
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
                    x = [0, 2]
                    y = [0 + k, 2 + k]
                print(x, y)
                data[i][j].append((x, y))
    plot(args.device, args.x_name, args.y_names, args.legends, args.settings,
         data)
    
