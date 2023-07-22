import math
import pickle
import numpy as np

class AlphaModel:
    def __init__(self, profiling_results) -> None:
        self.base_prefill = profiling_results[0]
        print(self.base_prefill)
    
    # load from .pkl file
    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        return cls(results)

    def get_latency(self, batch_size, seq_len):
        seq_len = math.ceil(seq_len / 32) * 32
        assert seq_len <= 1024
        if batch_size == 0: return 0
        # assert batch_size in self.base_prefill
        if batch_size in self.base_prefill:
            return self.base_prefill[batch_size][seq_len]
        elif batch_size == 1 and 2 in self.base_prefill:
            return self.base_prefill[2][seq_len]
        elif batch_size % 2 != 0 and batch_size - 1 in self.base_prefill and batch_size + 1 in self.base_prefill:
            return (self.base_prefill[batch_size - 1][seq_len] + self.base_prefill[batch_size + 1][seq_len]) / 2
        else:
            return np.Inf

class BetaModel:
    def __init__(self, profiling_results) -> None:
        self.base_prefill = profiling_results[0]
        self.adapter_prefill = profiling_results[1]
        print(self.adapter_prefill)
    
    # load from .pkl file
    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        return cls(results)
    
    def get_latency(self, rank_size, batch_size, seq_len):
        if rank_size == 0: return 0
        seq_len = math.ceil(seq_len / 32) * 32
        assert seq_len <= 1024
        if batch_size == 0: return 0
        # assert batch_size in self.base_prefill
        if batch_size in self.base_prefill:
            return self.adapter_prefill[rank_size][batch_size][seq_len] - self.base_prefill[batch_size][seq_len]
        elif batch_size == 1 and 2 in self.base_prefill:
            return self.adapter_prefill[rank_size][2][seq_len] - self.base_prefill[2][seq_len]
        elif batch_size % 2 != 0 and batch_size - 1 in self.base_prefill and batch_size + 1 in self.base_prefill:
            a = self.adapter_prefill[rank_size][batch_size - 1][seq_len] - self.base_prefill[batch_size - 1][seq_len]
            b = self.adapter_prefill[rank_size][batch_size + 1][seq_len] - self.base_prefill[batch_size + 1][seq_len]
            return (a + b) / 2
        else:
            return np.Inf
