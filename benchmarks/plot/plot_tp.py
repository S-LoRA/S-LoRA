import numpy as np
import matplotlib.pyplot as plt

# Sample data
labels = [["Llama-30B\n2xA100(40GB)\nn=10",  "Llama-30B\n2xA100(40GB)\nn=10"],
          ["Llama-30B\n2xA100(40GB)\nn=100", "Llama-30B\n4xA100(40GB)\nn=100"],
          ["Llama-70B\n2xA100(80GB)\nn=10",  "Llama-70B\n4xA100(80GB)\nn=10"]]
groups = [[[0.60, 3.29], [0.60, 2.73], [0.22, 1.81]],
          [[0.86, 4.09], [0.73, 3.42], [0.29, 2.41]],
          [[1.21, 5.09], [1.31, 5.12], [0.60, 3.46]]]

fig, ax = plt.subplots(1, 3)
FONTSIZE = 16

for i in range(3):
    label = labels[i]
    group = [groups[k][i] for k in range(3)]
    print(label, group)
    x = np.arange(len(label)) * 6  # the label locations
    print(x)
    width = 1.5
    
    rects1 = ax[i].bar(x - width, group[0], width, label="SLoRA")
    rects2 = ax[i].bar(x, group[1], width, label="SLoRA (w/o LoRA communication)")
    rects3 = ax[i].bar(x + width, group[2], width, label="SLoRA (base only)")
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(label, fontsize=FONTSIZE - 3)
    if i == 0:
        ax[i].set_ylabel("Throughput (req/s)", fontsize=FONTSIZE)
        ax[i].legend(loc=(-0.04, 1.1), ncols=3, fontsize=FONTSIZE)
    ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[i].set_axisbelow(True)
    fig.subplots_adjust(wspace=0.15)

fig.set_size_inches((12, 4))
fig.tight_layout()
fig.savefig("tp_results.pdf", bbox_inches="tight")
plt.show()
