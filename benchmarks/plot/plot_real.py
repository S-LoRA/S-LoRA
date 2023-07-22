import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


x_label_margin = {
        "Number of Adapters": 0.15,
        "CV Scale": 0.19,
        "Request Rate": 0.18,
        "Alpha": 0.195,
        }


def plot(device, x_name, y_names, legends, settings,
         data_set):
    FONTSIZE = 30
    figname = f"Real_workloads_on_{device.replace(' ', '_')}"

    fig, ax = plt.subplots(len(y_names), len(settings))
    for i, y_name in enumerate(y_names):
        for j, setting in enumerate(settings):
            curves = []
            for k, d in enumerate(data[i][j]):
                x, y = d
                curves.append(ax[i][j].plot(x, y, color=f"C{k}", linewidth=4,
                                            marker=".", markersize=FONTSIZE)[0])

            ax[i][j].grid(True, linestyle='-', linewidth=0.5,
                          alpha=0.5, color="black")
            
            #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            #ax.minorticks_on()
            #ax.grid(which='minor', linestyle=':', linewidth='', color='black')

            ax[i][j].set_xlim(0)
            ax[i][j].set_ylim(0)
            if i == len(y_names) - 1:
                left_margin = x_label_margin[x_name] + 0.2 * j
                vertical = 0.075
                fig.text(left_margin, vertical, x_name, va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            if i == 0:
                left_margin = 0.17 + 0.2 * j
                vertical = 0.9
                fig.text(left_margin, vertical, settings[j], va='center',
                         rotation='horizontal', fontsize=FONTSIZE)
            # ax[i][j].set_xlabel(x_name, fontsize=FONTSIZE)
            ax[i][j].tick_params(axis='both', which='major',
                                 labelsize=FONTSIZE, length=2, width=1)
            y_format = StrMethodFormatter("{x:.1f}")
            ax[i][j].yaxis.set_major_formatter(y_format)
            # ax.tick_params(axis='both', which='minor', length=5, width=1)

        left_margin = 0.085
        vertical = 0.81 - 0.16 * i
        fig.text(left_margin, vertical, y_name, va='center', rotation='vertical',
                 fontsize=FONTSIZE)
        fig.subplots_adjust(wspace=0.2)

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=len(legends) // min(2, (len(legends) - 1) // 4 + 1), fontsize=FONTSIZE)

    # margin = -0.01
    # if max(y) > 10:
    #     margin = -0.03
    # if max(y) > 100:
    #     margin = -0.04
    # Save figure
    fig.set_size_inches((40, 30))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="A10G (24GB)")
    parser.add_argument("--x-name", type=str, default="Request Rate")
    parser.add_argument("--y-names", type=str, action="append",
                        default=["Throughput (token/s)", "Average Latency (s)",
                                 "First Token Latency (s)", "SLO attainment",
                                 "User Satisfaction"])
    parser.add_argument("--legends", type=str, action="append",
                        default=["SLoRA", "SLoRA-PetS", "PEFT", "vllm-packed"])
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
                # TODO extract (x, y): (List, List)
                data[i][j].append(([0, 2], [0 + k, 2 + k]))
    plot(args.device, args.x_name, args.y_names, args.legends, args.settings,
         data)
    
