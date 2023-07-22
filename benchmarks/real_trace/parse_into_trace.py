import argparse
import random
import json

from tqdm import tqdm
from transformers import AutoTokenizer

model_mapping = {}
# copy from trace.py; (TODO) refactor
class Request:
    def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time):
        self.req_id = req_id
        self.model_dir = model_dir 
        self.adapter_dir = adapter_dir
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.req_time = req_time

    
    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, " \
               f"prompt_len={self.prompt_len}, output_len={self.output_len}, " \
               f"req_time={self.req_time}"

def downsample(json_file, ratio):
    with open(json_file, "r") as file:
       all_conversations = json.load(file)
    
    downsampled_conversations = []
    for conv in tqdm(all_conversations, desc="downsampling"):
        number = random.uniform(0, 1)
        if number < ratio:
            # accept this conversation
            downsampled_conversations.append(conv)
    print(f"Downsampled {len(downsampled_conversations)}")
    return downsampled_conversations 

def generate_model_mapping(conversations):
    global model_mapping
    num_rank_16 = 0
    num_rank_32 = 0
    for conv in conversations:
        model = conv["model"]
        # randomly choose between rank 16 or rank 32
        if model not in model_mapping.keys():
            if random.uniform(0, 1) < 0.5:
                rank = 16
                name = f"dummy-lora-7b-rank-{rank}-{num_rank_16}"
                num_rank_16 += 1
            else:
                rank = 32
                name = f"dummy-lora-7b-rank-{rank}-{num_rank_32}"
                num_rank_32 += 1
            model_mapping[model] = name    
    print(model_mapping)

def sort_and_rescale_by_req_time(conversations, duration):
    # sort first
    sorted_conversations = sorted(conversations, key=lambda d: d['tstamp']) 
    interval_start = sorted_conversations[0]["tstamp"]
    interval_end = sorted_conversations[-1]["tstamp"]
    # print(f"sorted time step: {[s['tstamp'] for s in sorted_conversations]}")

    for conv in conversations:
        tstamp = conv["tstamp"]
        assert interval_start <= tstamp and tstamp <= interval_end
        rescaled_tstamp = (tstamp - interval_start) / (interval_end - interval_start) + interval_start
        conv["tstamp"] = rescaled_tstamp
    return sorted_conversations 

def parse_into_req(conversations):
    global model_mapping
    model_dir = "/home/ubuntu/model_weights/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    reqs = []
    for idx, conv in enumerate(tqdm(conversations, desc="parse into reqs")):
        model = conv["model"]
        name = model_mapping[model]
        print(conv["conversation"][0]["content"])
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        
        req = Request(req_id=idx, model_dir=model_dir, adapter_dir=name, 
              prompt=conv["conversation"][0]["content"], prompt_len=prompt_len,
              output_len=output_len, req_time=conv["tstamp"])
        reqs.append(req)
    print(reqs)
    return reqs

if __name__ == "__main__":
    # For test purpose
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default="clean_chat_conv_20231016.json")
    parser.add_argument("--downsample-ratio", type=float, default=0.001)
    parser.add_argument("--duration", type=int, required=True)
    
    args = parser.parse_args()

    downsampled_conversations = downsample(args.in_file, args.downsample_ratio)
    generate_model_mapping(downsampled_conversations)
    sorted_conversations = sort_and_rescale_by_req_time(downsampled_conversations, args.duration)
    reqs = parse_into_req(sorted_conversations)
