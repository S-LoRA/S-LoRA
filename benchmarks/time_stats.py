import re


if __name__ == "__main__":
    with open("log", "r") as f:
        lines = f.readlines()

    load = 0
    prefetch = 0
    offload = 0
    prefill = 0
    decode = 0
    filter = 0
    filter_free_mem = 0
    load_scan = 0
    offload_free_mem = 0
    offload_torch_empty = 0
    offload_scan = 0
    offload_cat = 0
    load_A = 0
    load_B = 0
    for line in lines:
        match = re.search(r'-?\d+(\.\d+)?', line)
        if match is None: continue
        if "Function load_adapters took" in line:
            load += float(match.group(0))
        elif "Function offload_adapters took" in line:
            offload += float(match.group(0))
        elif "Function exposed_prefill_batch took" in line:
            prefill += float(match.group(0))
        elif "Function exposed_decode_batch took" in line:
            decode += float(match.group(0))
        elif "Function filter took" in line:
            filter += float(match.group(0))
        elif "Function load_lora_A took" in line:
            load_A += float(match.group(0))
        elif "Function load_lora_B took" in line:
            load_B += float(match.group(0))
        elif "cost realload" in line:
            load += float(match.group(0))
        elif "prefetch" in line:
            prefetch += float(match.group(0))
        elif "cost filter free mem manager" in line:
            filter_free_mem += float(match.group(0))
        elif "cost offload free mem manager" in line:
            offload_free_mem += float(match.group(0))
        elif "cost load scan" in line:
            load_scan += float(match.group(0))
        elif "cost offload torch.empty" in line:
            offload_torch_empty += float(match.group(0))
        elif "cost offload scan" in line:
            offload_scan += float(match.group(0))
        elif "cost offload cat" in line:
            offload_cat += float(match.group(0))

    print("load", load)
    print("prefetch", prefetch)
    print("offload", offload)
    print("prefill", prefill)
    print("decode", decode)
    print("filter", filter)
    print("filter_free_mem", filter_free_mem)
    print("offload_free_mem", offload_free_mem)
    print("offload_torch_empty", offload_torch_empty)
    print("offload_scan", offload_scan)
    print("offload_cat", offload_cat)
    print("load_scan", load_scan)
    print("load_A", load_A)
    print("load_B", load_B)
