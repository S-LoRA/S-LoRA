import torch
    

class MemoryManager:
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False):
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self._mem_cum_sum = torch.empty((size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
        self.always_copy = always_copy
    
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.key_buffer = [torch.empty((size, head_num, head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)]
        self.value_buffer = [torch.empty((size, head_num, head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)]
    
    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            print(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= len(select_index)
        return select_index
    
    def alloc_contiguous(self, need_size):
        if self.always_copy:
            return None
        if need_size > self.can_use_mem_size:
            print(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        sum_size = len(self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[need_size - 1:] - self._mem_cum_sum[0:sum_size - need_size + 1] + self.mem_state[0:sum_size - need_size + 1]
        can_used_loc = self.indexes[0:sum_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough cache to contiguous need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= len(select_index)
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end
    
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        self.can_use_mem_size += free_index.shape[0]
        self.mem_state[free_index] = 1
        if self.can_use_mem_size == len(self.mem_state):
            print(f"freed all gpu mem size {self.can_use_mem_size}")
        # print(f"free state {self.can_use_mem_size} all {len(self.mem_state)}")
        return
    
    def free_all(self):
        self.can_use_mem_size = len(self.mem_state)
        self.mem_state[:] = 1
