from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Dict, Any
import time

from slora.common.mem_allocator import MemoryAllocator
from slora.utils.infer_utils import calculate_time, mark_start, mark_end


@dataclass
class NaiveInferAdapter:
    adapter_dirs: List[str]  # all adapters on the server
    a_loc: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i
    a_start: torch.Tensor  # a_start[i] is the start location of adapter i
    a_len: torch.Tensor  # a_len[i] is the number of cells occupied by adapter i
    a_scaling: torch.Tensor  # a_scaling[i] is the scaling factor of adapter i
    idx_map: Dict[str, int]
    key_buffer: torch.Tensor
    value_buffer: torch.Tensor
    layer_num: int
    head_num: int
    head_dim: int

    @classmethod
    def init(cls, _layer_num, _head_num, _head_dim):
        return cls(
            adapter_dirs=[],
            a_loc=torch.empty(0, dtype=torch.long, device="cuda"),
            a_start=torch.empty(0, dtype=torch.long, device="cuda"),
            a_len=torch.empty(0, dtype=torch.long, device="cuda"),
            a_scaling=torch.empty(0, dtype=torch.float16, device="cuda"),
            idx_map={},
            key_buffer=[torch.empty(0, dtype=torch.float16, device="cuda")
                        for _ in range(_layer_num)],
            value_buffer=[torch.empty(0, dtype=torch.float16, device="cuda")
                          for _ in range(_layer_num)],
            layer_num=_layer_num,
            head_num=_head_num,
            head_dim=_head_dim,
        )


    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_A(self, adapter, start, end):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu()
            w_combined = adapter.layers[i].w_combined
            self.key_buffer[i][start:end] = w_combined[0]
            adapter.layers[i].offload_from_gpu()


    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_B(self, adapter, start, end):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu()
            w_combined = adapter.layers[i].w_combined
            self.value_buffer[i][start:end] = w_combined[1]
            adapter.layers[i].offload_from_gpu()


    # @calculate_time(show=True, min_cost_ms=0)
    def load_adapters(self, adapters, prefetch=False):
        assert prefetch is False
        if len(adapters) == 0:
            print(f"load 0 adapters, {len(self.adapter_dirs)} in total")
            return

        new_adapters = []
        rank_sum = 0
        for adapter in adapters:
            if adapter is not None and adapter.lora_dir not in self.idx_map:
                new_adapters.append(adapter)
                rank_sum += adapter.r * 4
        print(f"load {len(new_adapters)} adapters, {len(self.adapter_dirs) + len(new_adapters)} in total")

        if len(new_adapters) == 0:
            print(f"load 0 adapters, {len(self.adapter_dirs)} in total")
            return

        new_key_buffer = [torch.empty((rank_sum, self.head_num, self.head_dim), dtype=torch.float16, device="cuda")
                          for _ in range(self.layer_num)]
        new_value_buffer = [torch.empty((rank_sum, self.head_num, self.head_dim), dtype=torch.float16, device="cuda")
                            for _ in range(self.layer_num)]
        self.key_buffer = [torch.cat((self.key_buffer[i], new_key_buffer[i]))
                           for i in range(self.layer_num)]
        self.value_buffer = [torch.cat((self.value_buffer[i], new_value_buffer[i]))
                             for i in range(self.layer_num)]

        start_offset = self.a_start.shape[0]
        self.a_start = torch.cat((self.a_start, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        len_offset = self.a_len.shape[0]
        self.a_len = torch.cat((self.a_len, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        loc_offset = self.a_loc.shape[0]
        self.a_loc = torch.arange(0, self.a_loc.shape[0] + rank_sum, dtype=torch.long, device="cuda")

        cum_loc = loc_offset
        cum_loc_list = []
        for i, new_adapter in enumerate(new_adapters):
            cum_loc_list.append(cum_loc)
            self.idx_map[new_adapter.lora_dir] = len(self.adapter_dirs)
            self.adapter_dirs.append(new_adapter.lora_dir)
            self.a_start[start_offset + i] = cum_loc
            self.a_len[len_offset + i] = new_adapter.r * 4
            cum_loc += new_adapter.r * 4
        self.a_scaling = torch.cat((self.a_scaling, torch.tensor([adapter.scaling for adapter in new_adapters], dtype=torch.float16, device="cuda")))

        for i, new_adapter in enumerate(new_adapters):
            cum_loc = cum_loc_list[i]
            self.load_lora_A(new_adapter, cum_loc, cum_loc + new_adapter.r * 4)
            self.load_lora_B(new_adapter, cum_loc, cum_loc + new_adapter.r * 4)
   

    # @calculate_time(show=True, min_cost_ms=0)
    def offload_adapters(self, reserve_adapter_dirs):
        if len(reserve_adapter_dirs) == len(self.adapter_dirs):
            print(f"offload 0 adapters, {len(self.adapter_dirs)} remains")
            return
        if len(reserve_adapter_dirs) == 0:
            print(f"offload {len(self.adapter_dirs)} adapters, 0 remains")
            self.key_buffer=[torch.empty(0, dtype=torch.float16, device="cuda")
                             for _ in range(self.layer_num)]
            self.value_buffer=[torch.empty(0, dtype=torch.float16, device="cuda")
                               for _ in range(self.layer_num)]
            self.adapter_dirs=[]
            self.a_loc=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_start=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_len=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_scaling=torch.empty(0, dtype=torch.float16, device="cuda")
            self.idx_map={}
            return

        left_ind = []
        self.idx_map = {}
        new_adapter_dirs = []
        for i, adapter_dir in enumerate(self.adapter_dirs):
            if adapter_dir in reserve_adapter_dirs:
                left_ind.append(i)
                self.idx_map[adapter_dir] = len(new_adapter_dirs)
                new_adapter_dirs.append(adapter_dir)
        if len(new_adapter_dirs) == len(self.adapter_dirs):
            return
        print(f"offload {len(self.adapter_dirs) - len(left_ind)} adapters, "
              f"{len(left_ind)} remains")
        # left_ind = torch.tensor(left_ind, dtype=torch.int32, device="cuda")
        left_ind = torch.tensor(left_ind, dtype=torch.long, device="cuda")
        self.adapter_dirs = new_adapter_dirs
        rank_sum = torch.sum(self.a_len[left_ind]).item()
        
        new_a_len = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_start = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_scaling = torch.empty(len(left_ind), dtype=torch.float16, device="cuda")

        new_a_len[:] = self.a_len[left_ind]
        new_a_start[0] = 0
        new_a_start[1:] = torch.cumsum(new_a_len, dim=0)[:-1]
        new_a_scaling[:] = self.a_scaling[left_ind]

        # update self.key_buffer self.value_buffer
        new_key_buffer = [torch.empty((rank_sum, self.head_num, self.head_dim), dtype=torch.float16, device="cuda")
                          for _ in range(self.layer_num)]
        new_value_buffer = [torch.empty((rank_sum, self.head_num, self.head_dim), dtype=torch.float16, device="cuda")
                            for _ in range(self.layer_num)]
        copy_ind = torch.empty(rank_sum, dtype=torch.long, device="cuda")
        launch_var_len_copy_triton(self.a_start[left_ind], new_a_len,
                                   self.a_loc, new_a_start, copy_ind)
        new_key_buffer = [self.key_buffer[i][copy_ind] for i in range(self.layer_num)]
        new_value_buffer = [self.value_buffer[i][copy_ind] for i in range(self.layer_num)]
        self.key_buffer = new_key_buffer
        self.value_buffer = new_value_buffer

        self.a_len = new_a_len
        self.a_start = new_a_start
        self.a_loc = torch.arange(0, rank_sum, dtype=torch.long, device="cuda")
        self.a_scaling = new_a_scaling


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
