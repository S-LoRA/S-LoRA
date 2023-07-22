import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from slora.server.router.req_queue import ReqQueue
from slora.utils.infer_utils import  calculate_time


class ClusterReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, batch_num_adapters) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.batch_num_adapters = batch_num_adapters

    def _generate_new_batch_prioritizing_existing_adapters(self, current_batch:Batch, lora_ranks: dict[str, int]):
        filtered_waiting_req_list = list(filter(lambda req: req.adapter_dir in current_batch.adapter_dirs, self.waiting_req_list))
        request_ids_to_remove_from_waiting_queue = set()
        can_run_list = []
        new_batch_total_tokens = 0
        for idx, req in enumerate(filtered_waiting_req_list):
            if req.aborted:
                request_ids_to_remove_from_waiting_queue.add(req.request_id)
                continue
            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
                request_ids_to_remove_from_waiting_queue.add(req.request_id)
            else:
                break
        
        self.waiting_req_list = list(filter(lambda req: req.request_id not in request_ids_to_remove_from_waiting_queue, self.waiting_req_list))
        request_ids_to_remove_from_waiting_queue = set()

        # If filtered waiting list was not enough to max-out the current running batch, we resolve to FIFO
        for req in self.waiting_req_list:
            if req.aborted:
                request_ids_to_remove_from_waiting_queue.add(req.request_id)
                continue

            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
                request_ids_to_remove_from_waiting_queue.add(req.request_id)
            else:
                break
            
        self.waiting_req_list = list(filter(lambda req: req.request_id not in request_ids_to_remove_from_waiting_queue, self.waiting_req_list))

        return can_run_list

    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0

        for req in self.waiting_req_list:
            if req.aborted:
                aborted_count += 1
                continue

            if current_batch is not None and len(current_batch.adapter_dirs) >= self.batch_num_adapters:
                self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
                rest_of_batch = self._generate_new_batch_prioritizing_existing_adapters(current_batch, lora_ranks)
                can_run_list += rest_of_batch
                if len(can_run_list) != 0:
                    return Batch(uuid.uuid4().hex, can_run_list)
                else:
                    return None

            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        else:
            return None
