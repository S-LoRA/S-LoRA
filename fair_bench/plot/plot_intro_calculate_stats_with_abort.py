import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        json_file = json.load(f)
    
    total_time = json_file["result"]["total_time"]
    total_work  = 0
    for r in json_file["result"]["responses"]:
        if r["first_token_latency"] != -1:
            total_work += 256
    print(total_work/total_time)
    