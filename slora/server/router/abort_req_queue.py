import uuid
import asyncio
import numpy as np
import time
from typing import List
from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
from slora.server.router.req_queue import ReqQueue
from slora.utils.metric import attainment_func


class AbortReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.abort_req_list: List[str] = []
        self.req_time_stamp = []
        self.init_bs = 1
        self.apprx_req_rate = 1
        self.apprx_bs = self.init_bs
        self.last_req_num = 0
        self.last_time = time.time()
        
    def append(self, req):
        self.waiting_req_list.insert(0, req)
        self.req_time_stamp.insert(0, time.time())
        assert len(self.waiting_req_list) == len(self.req_time_stamp)
        return

    def reset_abort_list(self):
        self.abort_req_list = []

    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        new_batch_total_tokens = 0
        aborted_count = 0

        self.apprx_req_rate = int(0.7 * (len(self.waiting_req_list) - self.last_req_num) + 0.3 * self.apprx_req_rate)
        for i, req in enumerate(self.waiting_req_list):
            if attainment_func(time.time() - self.req_time_stamp[i] + 0.5) == 0:
                req.aborted = True
                aborted_count += 1
                abort_list.append(req)
                self.abort_req_list.append(req.request_id)
        self.req_time_stamp = [self.req_time_stamp[i] for i in range(len(self.req_time_stamp)) if self.waiting_req_list[i] not in abort_list]
        self.waiting_req_list = [req for req in self.waiting_req_list if req not in abort_list]
            
        if self.apprx_req_rate >= self.apprx_bs:
            print("apprx bs", self.apprx_bs, "req rate", self.apprx_req_rate)
            # choose from the latest requests
            for req in self.waiting_req_list:
                if req.aborted:
                    aborted_count += 1
                    abort_list.append(req)
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
        elif self.apprx_req_rate < self.apprx_bs:
            # choose from the earliest requests
            for req in reversed(self.waiting_req_list):
                if req.aborted:
                    aborted_count += 1
                    abort_list.append(req)
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
                
        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.req_time_stamp = [self.req_time_stamp[i] for i in range(len(self.req_time_stamp)) if self.waiting_req_list[i] not in can_run_list and self.waiting_req_list[i] not in abort_list]
            self.waiting_req_list = [req for req in self.waiting_req_list if req not in can_run_list and req not in abort_list]
            self.last_req_num = len(self.waiting_req_list)
            self.apprx_bs = max(int(0.7 * len(new_batch.reqs) + 0.3 * self.apprx_bs), self.init_bs)
            return new_batch
        else:
            return None
