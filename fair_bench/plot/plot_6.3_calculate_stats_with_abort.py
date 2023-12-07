import argparse
import json
import os
from matplotlib import pyplot as plt
FONTSIZE = 15

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    x_axis = ["5", "10", "15"]
    y_axis = []
    dirs = ["../LShare/rpm_5", "../LShare/rpm_10", "../LShare/rpm_15" ]
    for dir in dirs:
        with open(os.path.join(dir, "all_results_real.jsonl"), "r") as f:
            json_file = json.load(f)
        
        total_time = json_file["result"]["total_time"]
        total_work  = 0
        for r in json_file["result"]["responses"]:
            if r["first_token_latency"] != -1:
                total_work += r["output_len"]
        y_axis.append(total_work/total_time)
        print(total_work/total_time)
    
    # a threshold line
    plt.axhline(y=500, color='r', linestyle='--', label="VTC Throughput", marker="x", markersize=10)

    plt.plot(x_axis, y_axis, marker=".", label="RPM Throughput", markersize=10)
    plt.xlabel("Number of requests per minute threshold", fontsize=FONTSIZE)
    plt.ylabel("Throughput (Token/s)", fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    # set the font size of the x-axis and y-axis
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    
    plt.savefig("real_plots/LShare/LShare_overall_throughput.pdf", bbox_inches="tight")