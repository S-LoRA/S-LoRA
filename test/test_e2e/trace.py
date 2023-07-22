from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
import pickle
from typing import List, Tuple, Any

import numpy as np


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


def test_prompt(id):
    if id == 0:
        return "Rewrite and improve the following: Brown Suga Wellness know that you want to live a fulfilled life. To do that you need mental clarity and a well-crafted plan. The problem is you don\u2019t have time to evaluate hundreds of coaches and companies to piece together a wellness plan that bring you questionable results or don\u2019t cater to your unique needs as a woman of color which makes you feel stuck, overwhelmed, frustrated, invisible, dissatisfied, or unrepresented. We believe you deserve to be fulfilled in your life so we co-create your unique wellness plan. As a black entrepreneur & corporate woman I understand how you feel, I have been exactly where you are and know how frustrating it is to be overwhelmed, underrepresented and invisible without a plan to get better. This is why I am bringing over 20 year\u2019s experience as a Licensed Professional Counselor and Life Coach to help you create the life you want with a well-crafted wellness plan."
    elif id == 1:
        return "Paris is the capital city of"
    


def generate_requests(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed=42, id=1):
    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                test_prompt(id), int(input_lens[i]), int(output_lens[i]),
                                tic))
    return requests

