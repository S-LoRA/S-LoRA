import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os

COLOR = {"llama-7b": "C0",
         "llama-30b": "C1",
        }

MARKER = {"llama-7b": "x",
          "llama-30b": "^",
         }


def get_color(name):
    for key, value in COLOR.items():
        if name.startswith(key):
            return value
    return None


def get_marker(name):
    for key, value in MARKER.items():
        if name.startswith(key):
            return value
    return None


def plot(names, data, x_label, y_label, figname):
    FONTSIZE = 20

    legends = []
    curves = []
    fig, ax = plt.subplots()
    for i, (name, d) in enumerate(zip(names, data)):
        x, y = d
        curves.append(ax.plot(x, y, color=f"C{i}", marker=".", markersize=FONTSIZE)[0])
        legends.append(name)

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")
    
    #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    #ax.minorticks_on()
    #ax.grid(which='minor', linestyle=':', linewidth='', color='black')

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=15, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    # ax.tick_params(axis='both', which='minor', length=5, width=1)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=FONTSIZE)
    margin = -0.01
    if max(y) > 10:
        margin = -0.03
    if max(y) > 100:
        margin = -0.04
    fig.text(margin, 0.5, y_label, va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


def plot_two_y(names, x, y1, y2,
               x_label, y1_label, y2_label,
               figname):
    FONTSIZE = 10 

    legends = []
    curves = []

    fig, ax1 = plt.subplots()
    curves.append(ax1.plot(x, y1, color="C0", marker=".", markersize=FONTSIZE)[0])
    legends.append(names[0])
    ax1.set_xlabel(x_label, fontsize=FONTSIZE)
    ax1.set_ylabel(y1_label, color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    curves.append(ax2.plot(x, y2, color="C1", marker="x", markersize=FONTSIZE)[0])
    legends.append(names[1])
    ax2.set_ylabel(y2_label, color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    fig.legend(curves, legends, loc="upper center", ncol=len(legends), fontsize=FONTSIZE)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


def plot_two_y_multi(names_list, x, y_list,
                     x_label, y1_label, y2_label,
                     figname):
    FONTSIZE = 10 

    legends = []
    curves = []

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel(x_label, fontsize=FONTSIZE)
    ax1.set_ylabel(y1_label, color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    ax2.set_ylabel(y2_label, color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    for i, (names, ys) in enumerate(zip(names_list, y_list)):
        curves.append(ax1.plot(x[:len(ys[0])], ys[0], color=f"C{i*2}", marker=".", markersize=FONTSIZE)[0])
        legends.append(names[0])

        curves.append(ax2.plot(x[:len(ys[1])], ys[1], color=f"C{i*2+1}", linestyle="--", marker="x", markersize=FONTSIZE)[0])
        legends.append(names[1])

    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(legends) // min(4, len(legends)), fontsize=FONTSIZE)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

