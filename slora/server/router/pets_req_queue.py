import uuid
import asyncio
import numpy as np
from typing import List, Optional
from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
from slora.server.router.req_queue import ReqQueue


class PETSReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.alpha = None
        self.beta = None # will be set automatically in the profiling function in router.manager
        
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
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
    
    def intra_task_batching(self, lora_ranks):
        ## Preprocessing: gather the queries with the same adapter.
        clustered_queries_by_adapter = {}
        for query in self.waiting_req_list:
            adapter_dir = query.adapter_dir
            if adapter_dir in clustered_queries_by_adapter:
                clustered_queries_by_adapter[adapter_dir].append(query)
            else:
                clustered_queries_by_adapter[adapter_dir] = [query]

        ## DP
        mini_batches = []
        for adapter_dir, queries in clustered_queries_by_adapter.items():
            state_1st_stage = []
            split_idx_list = []

            ### Sort queries according to the sequence length in ascending order.
            queries = sorted(queries, key=lambda x: x.input_len)
            queries.insert(0, None)  # Sentinel.

            ### Initialize.
            state_1st_stage.append(0)
            split_idx_list.append(0)
            for j in range(1, len(queries)):
                min_cost = np.Inf  # INF
                split_idx = 0
                for k in range(1, j+1):
                    lora_rank  = lora_ranks[adapter_dir]
                    tmp = state_1st_stage[k-1] + self.beta.get_latency(lora_rank, j-k+1, queries[j].input_len)
                    if tmp < min_cost:
                        min_cost = tmp
                        split_idx = k-1
                split_idx_list.append(split_idx)
                state_1st_stage.append(min_cost)
            
            ### Split queries into mini-batches according to split_idx_list.
            
            end_idx = len(queries) - 1

            while(end_idx > 0):
                start_idx = split_idx_list[end_idx] + 1
                mini_batch = []
                max_len = queries[end_idx].input_len
                for j in range(start_idx, end_idx + 1):
                    mini_batch.append(queries[j])               
                mini_batches.append((mini_batch, max_len))
                end_idx = split_idx_list[end_idx]        
        
        return mini_batches
    
    # Inter-task batching.
    def inter_task_batching(self, mini_batches):
        ## Sort mini_batches according to the max sequence length.
        mini_batches = sorted(mini_batches, key=lambda x: x[1])
        mini_batches.insert(0, None)  # Sentinel.

        tmp = 0
        mini_batch_sum = [0]
        for mini_batch in mini_batches[1:]:
            tmp += len(mini_batch[0])
            mini_batch_sum.append(tmp)

        ## DP.
        state_2nd_stage = []
        split_idx_list = []
        state_2nd_stage.append(0)
        split_idx_list.append(0)

        for i in range(1, len(mini_batches)):
            min_cost = np.Inf  # INF
            split_idx = 0
            for j in range(1, i+1):
                total_samples = mini_batch_sum[i] - mini_batch_sum[j-1]
                tmp = state_2nd_stage[j-1] + self.alpha.get_latency(total_samples, mini_batches[i][1])
                if  tmp < min_cost:
                    min_cost = tmp
                    split_idx = j - 1
            split_idx_list.append(split_idx)
            state_2nd_stage.append(min_cost)

        ## Split mini_batches into final scheduled_batches.
        ### Split mini_batches into macro_batches.

        end_idx = len(mini_batches) - 1
        macro_batches = []
        while(end_idx > 0):
            start_idx = split_idx_list[end_idx] + 1
            macro_batch = []
            max_len = mini_batches[end_idx][1]
            for j in range(start_idx, end_idx + 1):
                macro_batch.append(mini_batches[j])               
            macro_batches.append((macro_batch, max_len))
            end_idx = split_idx_list[end_idx]        

        total_samples = 0
        for macro_batch in macro_batches:
             for mini_batch in macro_batch[0]:
                 total_samples += len(mini_batch[0])
        # print(total_samples)

        return macro_batches
    
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
        
        reqs = self.waiting_req_list
        # when waiting_reqs > 20
        if len(self.waiting_req_list) > 10:
            mini_batches = self.intra_task_batching(lora_ranks)
            macro_batches = self.inter_task_batching(mini_batches)
            
            macro_batch = macro_batches[-1][0]
            reqs = [req for minibatch in macro_batch for req in minibatch[0]]
            
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        for req in reqs:
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
            self.waiting_req_list = [req for req in self.waiting_req_list if req not in can_run_list and req not in abort_list]
            return new_batch
        else:
            return None


    def next_batch(self):
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            return next_batch
        else:
            return None
