import uuid
import asyncio
import numpy as np
import time
from typing import List
from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
from slora.server.router.req_queue import ReqQueue
from slora.utils.metric import attainment_func


class LShareReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.abort_req_list: List[str] = []
        self.req_time_stamp = []
        self.init_bs = 1
        self.apprx_req_rate = 1
        self.apprx_bs = self.init_bs
        self.last_req_num = 0
        self.last_time = time.time()
        self.rate_limit = 30 # per minute
        self.all_req_time_stamp = {}
        self.total_aborted = {}
        
    def append(self, req):
        # implement explicitly as admission control
        cur_req_time = time.time()
        if self.check_past_one_minute(req.adapter_dir, cur_req_time):
            req.aborted = True
            print(f"aborted {req.request_id}")
            self.abort_req_list.append(req.request_id)
            # aborted_count += 1
            # self.abort_req_list.append(req.request_id)
            return
        self.waiting_req_list.insert(0, req)
        self.req_time_stamp.insert(0, cur_req_time)
        if req.adapter_dir not in self.all_req_time_stamp.keys():
            self.all_req_time_stamp[req.adapter_dir] = []
        self.all_req_time_stamp[req.adapter_dir].insert(0, cur_req_time)
        assert len(self.waiting_req_list) == len(self.req_time_stamp)
        return

    def reset_abort_list(self):
        self.abort_req_list = []
    
    def check_past_one_minute(self, adapter, check_time):
        counter = 0
        if adapter not in self.all_req_time_stamp.keys():
            return False
        for _, req_time in enumerate(self.all_req_time_stamp[adapter]):
            # print(check_time - req_time)
            if check_time - req_time <= 60: # submitted in the last minute
                counter += 1
            if counter >= self.rate_limit:
                if adapter not in self.total_aborted.keys():
                    self.total_aborted[adapter] = 0
                self.total_aborted[adapter] += 1
                print(f"{adapter} aborted {self.total_aborted[adapter]} / {len(self.all_req_time_stamp[adapter])}")
                return True
                
        return False
        
    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        new_batch_total_tokens = 0
        aborted_count = 0

        # self.total = sum([len(v) for v in self.all_req_time_stamp.values()])
        # for i, req in enumerate(self.waiting_req_list):
        #    if self.check_past_one_minute(req.adapter_dir, self.req_time_stamp[i]):
        #        req.aborted = True
        #        aborted_count += 1
        #        abort_list.append(req)
        #        self.abort_req_list.append(req.request_id)
        # self.req_time_stamp = [self.req_time_stamp[i] for i in range(len(self.req_time_stamp)) if self.waiting_req_list[i] not in abort_list]
        # self.waiting_req_list = [req for req in self.waiting_req_list if req not in abort_list]
            
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
            return new_batch
        else:
            return None

    def update_counter(self, current_batch: Batch):
        pass