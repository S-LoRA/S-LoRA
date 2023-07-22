import gc
import torch


# TODO: will it slow down the program?
def suffix_cumsum(tensor, dim=-1, dtype=torch.int32):
    return torch.cumsum(tensor.flip(dim), dim, dtype=torch.int32).flip(dim)


class MemoryAllocator:
    def __init__(self, tot_size, cache_size, dtype, head_num, head_dim, layer_num):
        assert tot_size >= cache_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cell_size = head_num * head_dim

        self.tot_size = tot_size
        self.cache_size = cache_size

        self.reset_all_pool()


    def get_memory_size(self):
        dsize = 2 if self.dtype == torch.float16 else None
        return 2 * self.layer_num * self.tot_size * self.cell_size * dsize
  

    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= len(select_index)
        return select_index


    def alloc_contiguous(self, need_size):
        if need_size > self.can_use_mem_size:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[need_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - need_size + 1] + self.mem_state[0:self.tot_size - need_size + 1]
        can_used_loc = self.indexes[0:self.tot_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end


    def alloc_strip(self, need_block, block_size):
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[block_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - block_size + 1] + self.mem_state[0:self.tot_size - block_size + 1]
        loc_use = (loc_sums == block_size)
        torch.cumsum(loc_use, dim=0, dtype=torch.int32, out=loc_sums)

        block_start = torch.empty((loc_use.shape[0]), dtype=torch.int32, device="cuda")
        block_start[0] = loc_use[0]
        block_start[1:] = (loc_use[:-1] == 0) & (loc_use[1:] == 1)

        cum_max, _ = torch.cummax(block_start, dim=0)
        # (diff % block_size == 0) & loc_use
        mask = block_size - 1
        loc_use = (((loc_sums - cum_max) & mask) == 0) & loc_use
        can_use_loc = self.indexes[0:self.tot_size - block_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_block:
            raise Exception(f"no enough pool space for alloc_strip, "
                            f"need {need_block} blocks, {can_use_loc.shape[0]} left")
        can_use_loc = can_use_loc[:need_block]
        select_index = torch.empty((block_size, need_block), dtype=torch.int32, device="cuda")
        for i in range(block_size):
            select_index[i] = can_use_loc + i
        select_index = select_index.T.reshape(-1)

        self.mem_state[select_index] = 0
        self.can_use_mem_size -= select_index.shape[0]
        return select_index


    def alloc_grid(self, need_grid, grid_size):
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[grid_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - grid_size + 1] + self.mem_state[0:self.tot_size - grid_size + 1]
        loc_use = (loc_sums == grid_size)

        mask = grid_size - 1
        loc_use = ((self.indexes[:self.tot_size - grid_size + 1] & mask) == 0) & loc_use
        can_use_loc = self.indexes[0:self.tot_size - grid_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_grid:
            raise Exception(f"no enough pool space for alloc_strip, "
                            f"need {need_grid} grids, {can_use_loc.shape[0]} left")
        can_use_loc = can_use_loc[:need_grid]
        select_index = torch.empty((grid_size, need_grid), dtype=torch.int32, device="cuda")
        for i in range(grid_size):
            select_index[i] = can_use_loc + i
        select_index = select_index.T.reshape(-1)

        self.mem_state[select_index] = 0
        self.can_use_mem_size -= select_index.shape[0]
        return select_index


    def alloc_prefix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_prefix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_prefix}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size_prefix -= len(select_index)
        return select_index
    

    def alloc_contiguous_prefix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_prefix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_prefix}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[need_size - 1:self.cache_size] - self._mem_cum_sum[0:self.cache_size - need_size + 1] + self.mem_state[0:self.cache_size - need_size + 1]
        can_used_loc = self.indexes[0:self.cache_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size_prefix}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size_prefix -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end


    def alloc_suffix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_suffix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        
        self._mem_cum_sum = suffix_cumsum(self.mem_state, dim=0, dtype=torch.int32)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size_suffix -= len(select_index)
        return select_index
    

    def alloc_contiguous_suffix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_suffix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        
        self._mem_cum_sum = suffix_cumsum(self.mem_state, dim=0, dtype=torch.int32)
        assert len(self._mem_cum_sum) == self.cache_size
        loc_sums = (self._mem_cum_sum[0:self.cache_size - need_size + 1] - self._mem_cum_sum[need_size - 1:] +
                    self.mem_state[need_size - 1:])
        can_used_loc = self.indexes[0:self.cache_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size_suffix -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end
 
    
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        self.can_use_mem_size += free_index.shape[0]
        # self.can_use_mem_size_prefix += torch.sum(free_index < self.cache_size)
        # self.can_use_mem_size_suffix += torch.sum(free_index >= self.cache_size)
        self.mem_state[free_index] = 1

        # if self.can_use_mem_size_prefix + self.can_use_mem_size_suffix == self.tot_size:
        #     print(f"freed all gpu mem size {self.tot_size}")
        # print(f"free state {self.can_use_mem_size_prefix} + {self.can_use_mem_size_suffix} all {self.tot_size}")
        return
    
    def free_all(self):
        self.mem_state[:] = 1
        self.can_use_mem_size = self.tot_size
        # self.can_use_mem_size_prefix = self.cache_size
        # self.can_use_mem_size_suffix = self.tot_size - self.cache_size
    

    def delete_all_pool(self):
        self.mem_state = None
        self._mem_cum_sum = None
        self.indexes = None
        self.can_use_mem_size = 0
        # self.can_use_mem_size_prefix = 0
        # self.can_use_mem_size_suffix = 0
        self.buffer = None
        gc.collect()

    def delete_all_cache(self):
        self.delete_all_pool()


    def reset_all_pool(self):
        self.mem_state = torch.ones((self.tot_size,), dtype=torch.bool, device="cuda")
        self._mem_cum_sum = torch.empty((self.tot_size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, self.tot_size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = self.tot_size
        # self.can_use_mem_size_prefix = self.cache_size
        # self.can_use_mem_size_suffix = self.tot_size - self.cache_size
        self.key_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                       dtype=self.dtype, device="cuda")
                           for _ in range(self.layer_num)]
        self.value_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                       dtype=self.dtype, device="cuda")
                           for _ in range(self.layer_num)]
 

    def reset_all_cache(self):
        self.reset_all_pool()
