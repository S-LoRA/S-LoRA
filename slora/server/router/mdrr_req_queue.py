import asyncio
import uuid
from collections import deque
from typing import List, Optional

import numpy as np

from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
from slora.server.router.req_queue import ReqQueue


class MDRRReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 adapter_dirs, fair_weights,
                 input_price=1, output_price=2) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.input_price = input_price
        self.output_price = output_price
        self.dcounter = {}
        self.user_req_list = {}
        self.active_list = deque()
        self.quantum = max_total_tokens / 20
        # record last oom adapter
        self.last_oom_adapter = None

        self.adapter_dirs = adapter_dirs
        self.fair_weights = fair_weights

        self.fairw = {}
        for i in range(len(adapter_dirs)):
            if i < len(fair_weights):
                self.fairw[adapter_dirs[i]] = fair_weights[i]
            else:
                self.fairw[adapter_dirs[i]] = 1
        
        
    def append(self, req):
        self.waiting_req_list.append(req)
        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
            self.dcounter[req.adapter_dir] = 0
        else:
            self.user_req_list[req.adapter_dir].append(req)

        # waiting queue was empty before and not in active list
        if len(self.user_req_list[req.adapter_dir]) == 1 and req.adapter_dir not in self.active_list:
            # insert into active list
            self.active_list.append(req.adapter_dir)

    
    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                           req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False


    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        if len(self.dcounter) == 0:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        OOM = False
        while True:
            if len(self.dcounter) == 0:
                break
            if len(self.active_list) == 0:
                break
            # pop the first adapter in active list
            adapter_dir = self.active_list.popleft()
            # if last oom adapter's dcounter less than 0, skip it and add it to active list
            if adapter_dir == self.last_oom_adapter:
                if self.dcounter[adapter_dir] < 0:
                    self.active_list.append(adapter_dir)
                    self.last_oom_adapter = None
                    continue
            #  add quantum if dcounter less than 0
            if self.dcounter[adapter_dir] <= 0:
                self.dcounter[adapter_dir] += self.quantum
            while self.dcounter[adapter_dir] > 0 and len(self.user_req_list[adapter_dir]) > 0:
                req = self.user_req_list[adapter_dir][0]
                if req.aborted:
                    aborted_count += 1
                    abort_list.append(req)
                    self.user_req_list[adapter_dir].popleft()
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                    self.user_req_list[adapter_dir].popleft()
                    # update dcounter
                    self.dcounter[adapter_dir] -= req.input_len * self.input_price / self.fairw[adapter_dir]
                else:
                    # insert at the head of active list
                    self.active_list.appendleft(adapter_dir)
                    self.last_oom_adapter = adapter_dir
                    OOM = True
                    break
            if OOM:
                break
            if len(self.user_req_list[adapter_dir]) != 0:
                self.active_list.append(adapter_dir)
            elif self.dcounter[adapter_dir] < 0:
                self.active_list.append(adapter_dir)

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = [req for req in self.waiting_req_list
                                     if req not in can_run_list and req not in abort_list]
            return new_batch
        else:
            return None

    
    def update_counter(self, current_batch: Batch):
        for req in current_batch.reqs:
            self.dcounter[req.adapter_dir] -= 1 * self.output_price / self.fairw[req.adapter_dir]


    def next_batch(self):
        raise NotImplementedError()
