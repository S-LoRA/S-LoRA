from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Dict, Any
import time

from slora.common.mem_allocator import MemoryAllocator
from slora.utils.infer_utils import calculate_time, mark_start, mark_end


@dataclass
class InferAdapter:
    adapter_dirs: List[str]  # all adapters on the server
    a_loc: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i
    a_start: torch.Tensor  # a_start[i] is the start location of adapter i
    a_len: torch.Tensor  # a_len[i] is the number of cells occupied by adapter i
    a_scaling: torch.Tensor  # a_scaling[i] is the scaling factor of adapter i
    mem_manager: MemoryAllocator

    idx_map: Dict[str, int]
    prefetch_tag: Dict[str, int]
    cur_tag: int

    prefetch_stream: Any

    @classmethod
    def init(cls, mem_manager, prefetch_stream):
        return cls(
            adapter_dirs=[],
            a_loc=torch.empty(0, dtype=torch.long, device="cuda"),
            a_start=torch.empty(0, dtype=torch.long, device="cuda"),
            a_len=torch.empty(0, dtype=torch.long, device="cuda"),
            a_scaling=torch.empty(0, dtype=torch.float16, device="cuda"),
            mem_manager=mem_manager,
            idx_map={},
            prefetch_tag={},
            cur_tag=0,
            prefetch_stream=prefetch_stream,
        )


    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_A(self, adapter, loc, prefetch=False):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        head_num = adapter.network_config["num_attention_heads"]
        head_dim = h // head_num

        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu(prefetch=prefetch)
            #self.mem_manager.key_buffer[i][loc[:r]] = adapter.layers[i].q_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r:r * 2]] = adapter.layers[i].k_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 2:r * 3]] = adapter.layers[i].v_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 3:r * 4]] = adapter.layers[i].o_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)

            w_combined = adapter.layers[i].w_combined
            self.mem_manager.key_buffer[i][loc] = w_combined[0]

            #self.mem_manager.key_buffer[i][loc[:r]] = w_combined[0].T.reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r:r * 2]] = w_combined[1].T.reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 2:r * 3]] = w_combined[2].T.reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 3:r * 4]] = w_combined[3].T.reshape(r, head_num, head_dim)

            adapter.layers[i].offload_from_gpu()


    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_B(self, adapter, loc, prefetch=False):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        head_num = adapter.network_config["num_attention_heads"]
        head_dim = h // head_num
        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu(prefetch=prefetch)
            # this copy on gpu takes very few time, ~3ms for the following lines of copy
            #self.mem_manager.value_buffer[i][loc[:r]] = adapter.layers[i].q_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r:r * 2]] = adapter.layers[i].k_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 2:r * 3]] = adapter.layers[i].v_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 3:r * 4]] = adapter.layers[i].o_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)

            w_combined = adapter.layers[i].w_combined
            self.mem_manager.value_buffer[i][loc] = w_combined[1]

            #self.mem_manager.value_buffer[i][loc[:r]] = w_combined[4].reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r:r * 2]] = w_combined[5].reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 2:r * 3]] = w_combined[6].reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 3:r * 4]] = w_combined[7].reshape(r, head_num, head_dim)

            adapter.layers[i].offload_from_gpu()

    # @calculate_time(show=True, min_cost_ms=0)
    def load_adapters(self, adapters, prefetch=False):
        # func_name = "realload" if not prefetch else "prefetch"
        # mark_start(func_name)
        if len(adapters) == 0:
            print(f"load 0 adapters, {len(self.adapter_dirs)} in total")
            return

        if prefetch:
            self.cur_tag ^= 1
            capacity = self.mem_manager.can_use_mem_size
            new_adapters = []
            tot_size = 0
            # mark_start("load scan")
            for adapter in adapters:
                self.prefetch_tag[adapter.lora_dir] = self.cur_tag
                if adapter is not None and adapter.lora_dir not in self.idx_map:
                    if tot_size + adapter.r * 4 > capacity:
                        break
                    new_adapters.append(adapter)
                    tot_size += adapter.r * 4
            # mark_end("load scan")
            print(f"prefetch {len(new_adapters)} adapters, "
                  f"{len(self.adapter_dirs) + len(new_adapters)} in total")
        else:
            new_adapters = []
            tot_size = 0
            # mark_start("load scan")
            for adapter in adapters:
                if adapter is not None and adapter.lora_dir not in self.idx_map:
                    new_adapters.append(adapter)
                    tot_size += adapter.r * 4
            # mark_end("load scan")
            print(f"load {len(new_adapters)} adapters, {len(self.adapter_dirs) + len(new_adapters)} in total")

        new_loc = self.mem_manager.alloc(tot_size)
        # assert len(new_loc) == tot_size
        start_offset = self.a_start.shape[0]
        self.a_start = torch.cat((self.a_start, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        len_offset = self.a_len.shape[0]
        self.a_len = torch.cat((self.a_len, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        loc_offset = self.a_loc.shape[0]
        self.a_loc = torch.cat((self.a_loc, torch.empty(tot_size, dtype=torch.long, device="cuda")))

        cum_loc = 0
        cum_loc_list = []
        for i, new_adapter in enumerate(new_adapters):
            cum_loc_list.append(cum_loc)
            self.idx_map[new_adapter.lora_dir] = len(self.adapter_dirs)
            self.adapter_dirs.append(new_adapter.lora_dir)
            self.a_start[start_offset + i] = loc_offset + cum_loc
            self.a_len[len_offset + i] = new_adapter.r * 4
            self.a_loc[loc_offset + cum_loc: loc_offset + cum_loc + new_adapter.r * 4] = (
                    new_loc[cum_loc: cum_loc + new_adapter.r * 4])
            cum_loc += new_adapter.r * 4
        self.a_scaling = torch.cat((self.a_scaling, torch.tensor([adapter.scaling for adapter in new_adapters], dtype=torch.float16, device="cuda")))

        #if prefetch:
        #    torch.cuda.synchronize()
        #    tic1 = time.time()

        if prefetch:
            with torch.cuda.stream(self.prefetch_stream):
                new_loc = new_loc.clone()
                for i, new_adapter in enumerate(new_adapters):
                    #self.idx_map[new_adapter.lora_dir] = len(self.adapter_dirs)
                    #self.adapter_dirs.append(new_adapter.lora_dir)
                    #self.a_start[start_offset + i] = loc_offset + cum_loc
                    #self.a_len[len_offset + i] = new_adapter.r * 4

                    cum_loc = cum_loc_list[i]
                    self.load_lora_A(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)
                    self.load_lora_B(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)

                    #self.load_lora_A(new_adapter, None, prefetch)
                    #self.load_lora_B(new_adapter, None, prefetch)
        else:
            for i, new_adapter in enumerate(new_adapters):
                cum_loc = cum_loc_list[i]
                self.load_lora_A(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)
                self.load_lora_B(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)

            #if prefetch:
        #    tic2 = time.time()
        #    torch.cuda.synchronize()
        #    tic3 = time.time()
        #    print("launch time", tic2 - tic1, flush=True)
        #    print("total time", tic3 - tic1, flush=True)
        # mark_end(func_name)
        # print(f"current adapters on batch (loaded {len(new_adapters)})",
        #       len(self.adapter_dirs), self.adapter_dirs)
        # print(self.mem_manager.can_use_mem_size_suffix // 4 / 32)
    

    # @calculate_time(show=True, min_cost_ms=0)
    def offload_adapters(self, reserve_adapter_dirs):
        if len(reserve_adapter_dirs) == len(self.adapter_dirs):
            print(f"offload 0 adapters, {len(self.adapter_dirs)} remains")
            return
        if len(reserve_adapter_dirs) == 0:
            print(f"offload {len(self.adapter_dirs)} adapters, 0 remains")
            self.mem_manager.free(self.a_loc)
            self.adapter_dirs=[]
            self.a_loc=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_start=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_len=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_scaling=torch.empty(0, dtype=torch.float16, device="cuda")
            self.idx_map={}
            return

        # mark_start("offload scan")
        remove_ind = []
        left_ind = []
        new_adapter_dirs = []
        self.idx_map = {}
        for i, adapter_dir in enumerate(self.adapter_dirs):
            if (adapter_dir not in reserve_adapter_dirs and
                (adapter_dir not in self.prefetch_tag or
                 self.prefetch_tag[adapter_dir] != self.cur_tag)):
                remove_ind.append(self.a_loc[self.a_start[i]:self.a_start[i] + self.a_len[i]])
            else:
                left_ind.append(i)
                self.idx_map[adapter_dir] = len(new_adapter_dirs)
                new_adapter_dirs.append(adapter_dir)
        if len(remove_ind) == 0:
            return
        # mark_end("offload scan")
        self.adapter_dirs = new_adapter_dirs
        tot_size = torch.sum(self.a_len[left_ind]).item()
        print(f"offload {len(remove_ind)} adapters, {len(left_ind)} remains")

        # mark_start("offload cat")
        remove_ind = torch.cat(remove_ind)
        # mark_end("offload cat")
        # release memory
        # mark_start("offload free mem manager")
        self.mem_manager.free(remove_ind)
        # mark_end("offload free mem manager")
        
        # reset indexing
        # mark_start("offload torch.empty")
        new_a_len = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_start = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_scaling = torch.empty(len(left_ind), dtype=torch.float16, device="cuda")
        new_a_loc = torch.empty(tot_size, dtype=torch.long, device="cuda")
        # mark_end("offload torch.empty")

        new_a_len[:] = self.a_len[left_ind]
        new_a_start[0] = 0
        new_a_start[1:] = torch.cumsum(new_a_len, dim=0)[:-1]
        new_a_scaling[:] = self.a_scaling[left_ind]
        # mark_start("offload a_loc update")
        launch_var_len_copy_triton(self.a_start[left_ind], new_a_len,
                                   self.a_loc, new_a_start, new_a_loc)
        # mark_end("offload a_loc update")

        self.a_start = new_a_start
        self.a_len = new_a_len
        self.a_loc = new_a_loc
        self.a_scaling = new_a_scaling

        # print(f"current adapters on batch (offloaded {len(remove_ind)})",
        #       len(self.adapter_dirs), self.adapter_dirs)
        # print(self.mem_manager.can_use_mem_size_suffix // 4 / 32)


import triton
import triton.language as tl


@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=new_offset < length)


def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)

    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, new_a_start, new_a_location, BLOCK_SIZE)


"""
from cupyx import jit
import cupy
import torch


@jit.rawkernel()
def var_len_copy_kernel(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                        BLOCK_SIZE):
    a_id = jit.blockIdx.x
    t_id = jit.threadIdx.x
    for i in range(t_id, old_a_len[a_id], BLOCK_SIZE):
        new_a_location[new_a_start[a_id] + i] = old_a_location[old_a_start[a_id] + i]


def launch_var_len_copy(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 128
    print(BLOCK_SIZE)
    grid_size = (len(old_a_start),)
    assert len(old_a_start) == len(new_a_start) == len(old_a_len)
    block_size = (BLOCK_SIZE,)

    var_len_copy_kernel(grid_size, block_size,
        (cupy.asarray(old_a_start),
         cupy.asarray(old_a_len),
         cupy.asarray(old_location),
         cupy.asarray(new_a_start),
         cupy.asarray(new_a_location),
         BLOCK_SIZE))
"""
