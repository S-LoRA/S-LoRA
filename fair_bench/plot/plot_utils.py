import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os


def plot(names, x, ys, x_label, y_label, figname):
    FONTSIZE = 15

    legends = []
    curves = []
    fig, ax = plt.subplots()
    for i, (name, y) in enumerate(zip(names, ys)):
        curves.append(ax.plot(x, y, color=f"C{i}", marker=".", markersize=FONTSIZE)[0])
        legends.append(name)

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=15, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(0.5, 1),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=FONTSIZE)
    fig.text(0, 0.5, y_label, va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((12, 8))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")
