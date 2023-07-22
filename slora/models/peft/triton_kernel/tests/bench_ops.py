import argparse
import itertools

import torch

import slora._kernels
from slora.models.peft.triton_kernel.lora.lora_decode import lora_get_qkvo_fwd_expand, lora_get_qkvo_fwd_shrink

from benchmark_utils import bench, gc_torch

import numpy as np
import sys
from typing import List, Tuple
import random
random.seed(10)

import cutlass_bindings
import cutlass.backend as pycutlass
from cutlass.backend import *
from cutlass.backend.utils.device import device_cc


def assert_close(a, b):
  rtol, atol = {
      torch.float16: (5e-3, 5e-3),
      torch.bfloat16: (3e-2, 2e-2),
  }[a.dtype]
  torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

# Check that the device is of a sufficient compute capability
cc = device_cc()
assert cc >= 70, "The CUTLASS Python grouped GEMM example requires compute capability greater than or equal to 70."

np.random.seed(0)

# Allocate a pool of device memory to be used by the kernel
pycutlass.get_memory_pool(init_pool_size=2**30, max_pool_size=2**32)

# Set the compiler to use to NVCC
pycutlass.compiler.nvcc()

# Set up A, B, C and accumulator
alignment = 1
A = TensorDescription(cutlass_bindings.float16, cutlass_bindings.RowMajor, alignment)
B = TensorDescription(cutlass_bindings.float16, cutlass_bindings.ColumnMajor, alignment)
C = TensorDescription(cutlass_bindings.float16, cutlass_bindings.RowMajor, alignment)
element_acc = cutlass_bindings.float32
element_epilogue = cutlass_bindings.float16

# Select instruction shape based on the Tensor Core instructions supported
# by the device on which we are running
if cc == 70:
    instruction_shape = [8, 8, 4]
elif cc == 75:
    instruction_shape = [16, 8, 8]
else:
    # Use CUTLASS kernels for CC 80 by default (e.g., for cases in which SM86 is used)
    cc = 80
    instruction_shape = [16, 8, 16]

math_inst = MathInstruction(
    instruction_shape,
    A.element, B.element, element_acc,
    cutlass_bindings.OpClass.TensorOp,
    MathOperation.multiply_add
)

tile_description = TileDescription(
    [128, 128, 32],   # Threadblock shape
    2,                # Number of stages
    [2, 2, 1],        # Number of warps within each dimension of the threadblock shape
    math_inst
)

epilogue_functor = pycutlass.LinearCombination(C.element, C.alignment, element_acc, element_epilogue)

operation = GemmOperationGrouped(
    arch=cc, tile_description=tile_description,
    A=A, B=B, C=C,
    epilogue_functor=epilogue_functor,
    precompute_mode=SchedulerMode.Device)

operations = [operation, ]

# Compile the operation
pycutlass.compiler.add_module(operations)

class batch_loraA_param:
  def __init__(
      self,
      num_layers: int,
      in_features: int,
      out_features: int,
      lora_rank: int,
      lora_num: int,
      per_lora: int,
      dtype: str,
      device: torch.device,
  ):
    dtype = getattr(torch, dtype)
    self.request_bins = torch.randint(per_lora, per_lora+1, (lora_num,), dtype=torch.long, device=device)
    batch_size = sum(self.request_bins)
    self.x = torch.randn((batch_size, in_features), dtype=dtype, device=device)
    self.y = torch.zeros((batch_size, out_features), dtype=dtype, device=device)
    self.y_ref = torch.zeros((batch_size, out_features), dtype=dtype, device=device)
    
    self.w_T_all = torch.randn(
        (16384, 1, 1, in_features),
        dtype=dtype,
        device=device)
    self.w_T_deduplicate = torch.randn(
        (lora_num, num_layers, out_features, in_features),
        dtype=dtype,
        device=device)
    # self.indicies = torch.arange(0, batch_size, dtype=torch.long, device=device)
    self.indicies = torch.repeat_interleave(torch.arange(0, lora_num, dtype=torch.long, device=device), self.request_bins)
    
    values = torch.arange(0, 16384, dtype=torch.long, device=device)
    values = values[torch.randperm(len(values))]
    self.cell_indicies = values[:lora_num * out_features]
    for idx, lora in enumerate(self.w_T_deduplicate):
        for i in range(out_features):
            self.w_T_all[self.cell_indicies[idx*out_features+i],0,0,:].copy_(lora[0,i,:])
    self.start_indicies = torch.arange(0, len(self.cell_indicies), out_features, dtype=torch.long, device=device)
    self.lora_ranks = torch.tensor([lora_rank*4]*lora_num, dtype=torch.long, device=device)
    self.scale = torch.tensor([1.0]*lora_num, dtype=torch.float16, device=device)

class batch_loraB_param:
  def __init__(
      self,
      num_layers: int,
      in_features: int,
      out_features: int,
      lora_rank: int,
      lora_num: int,
      per_lora: int,
      dtype: str,
      device: torch.device,
  ):
    dtype = getattr(torch, dtype)
    self.request_bins = torch.randint(per_lora, per_lora+1, (lora_num,), dtype=torch.long, device=device)
    batch_size = sum(self.request_bins)
    self.x = torch.randn((batch_size, in_features), dtype=dtype, device=device)
    self.y = torch.zeros((batch_size, out_features), dtype=dtype, device=device)
    self.y_ref = torch.zeros((batch_size, out_features), dtype=dtype, device=device)
    
    self.w_T_all = torch.randn(
        (16384, 1, 1, out_features),
        dtype=dtype,
        device=device)
    self.w_T_deduplicate = torch.randn(
        (lora_num, num_layers, out_features, in_features),
        dtype=dtype,
        device=device)
    # self.indicies = torch.arange(0, batch_size, dtype=torch.long, device=device)
    self.indicies = torch.repeat_interleave(torch.arange(0, lora_num, dtype=torch.long, device=device), self.request_bins)
    
    values = torch.arange(0, 16384, dtype=torch.long, device=device)
    values = values[torch.randperm(len(values))]
    self.cell_indicies = values[:lora_num * lora_rank]
    for idx, lora in enumerate(self.w_T_deduplicate):
        for i in range(lora_rank):
            self.w_T_all[self.cell_indicies[idx*lora_rank+i],0,0,:].copy_(lora.view(lora_rank, out_features)[i])

    self.start_indicies = torch.arange(0, len(self.cell_indicies), lora_rank, dtype=torch.long, device=device)
    self.lora_ranks = torch.tensor([lora_rank*4]*lora_num, dtype=torch.long, device=device)
    self.scale = torch.tensor([1.0]*lora_num, dtype=torch.float16, device=device)

def gemm(operation, arguments):
        operation.run(arguments)
        # arguments.sync()

def bench_ggemm_A():
  lora_ranks = [8, 16, 32, 64, 128]
  weight_sizes = [
      768,
      1024,
      2048,
      2560,
      3072,
      4096,
      5120,
      7168,
      8192,
      9216,
      10240,
      11008,
      12288,
      13824,
      16384,
      20480,
      28672,
      36864,
      49152,
  ]
  # combo = [(1, 64), (4, 16), (4, 64), (16, 4), (16, 16), (16, 64), (64, 1), (64, 4), (64, 16)]
  combo = [(1,512)]
  weight_size = 4096
  device = torch.device("cuda:0")

  print("bench_ggemm")
  # for (in_features,
  #      out_features), batch_size in itertools.product(all_sizes, batch_sizes):
  for out_features, (lora_num, per_lora) in itertools.product(lora_ranks, combo):
    torch.manual_seed(0xabcdabcd987)
    try:
      gc_torch()
      t = batch_loraA_param(
          num_layers=1,
          in_features=weight_size,
          out_features=out_features,
          lora_rank = out_features,
          lora_num=lora_num,
          per_lora=per_lora,
          dtype="float16",
          device=device,
      )
      outputs = [
        f"h1={weight_size}",
        f"h2={out_features}",
        f"lora_num={lora_num}",
        f"per_lora={per_lora}",
        f"bs={sum(t.request_bins):2d}"
    ]
      layer_idx = 0
      scale = 1.0
      alpha = 1
      beta = 0

      problem_sizes = []
      tensor_As = []
      tensor_Bs = []
      tensor_Cs = []
      tensor_Ds = []
      cur_pos = 0
      # Initialize tensors for each problem in the group
      for idx, req_num in enumerate(t.request_bins):
        problem_sizes.append(cutlass_bindings.gemm.GemmCoord(req_num, out_features, weight_size))
        tensor_As.append(t.x[cur_pos:cur_pos+req_num])
        tensor_Bs.append(t.w_T_deduplicate[idx])
        tensor_Cs.append(t.y[cur_pos:cur_pos+req_num])
        tensor_Ds.append(t.y[cur_pos:cur_pos+req_num])
        cur_pos += req_num
        idx += 1
      
      arguments = GemmGroupedArguments(
          operation, problem_sizes, tensor_As, tensor_Bs, tensor_Cs, tensor_Ds,
          output_op=operation.epilogue_type(alpha, beta)
      )
      # test correctness
      gemm(operation, arguments)
      arguments.sync()
      slora._kernels.dispatch_bgmv(
          t.y_ref, t.x, t.w_T_all.view(-1, 32, 4096//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale)
      assert_close(t.y_ref, t.y)

      result = bench(lambda: gemm(operation, arguments))
      outputs.append(f"\n gemm: {result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")

      result2 = bench(lambda: slora._kernels.dispatch_bgmv(
          t.y_ref, t.x, t.w_T_all.view(-1, 32, 4096//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale))
      outputs.append(f"\n bgmv: {result2.avg()*1e6:3.0f}us±{result2.std()*1e6:3.0f}us")
    except torch.cuda.OutOfMemoryError:
      outputs.append("OOM")

    print(" | ".join(outputs))

def bench_ggemm_B():
  lora_ranks = [8, 16, 32, 64, 128]
  # combo = [(1, 64), (4, 16), (4, 64), (16, 4), (16, 16), (16, 64), (64, 1), (64, 4), (64, 16)]
  combo = [(1,512)]
  weight_size = 4096
  device = torch.device("cuda:0")

  print("bench_ggemm_B")
  # for (in_features,
  #      out_features), batch_size in itertools.product(all_sizes, batch_sizes):
  for in_features, (lora_num, per_lora) in itertools.product(lora_ranks, combo):
    torch.manual_seed(0xabcdabcd987)
    try:
      gc_torch()
      t = batch_loraB_param(
          num_layers=1,
          in_features=in_features,
          out_features=weight_size,
          lora_rank = in_features,
          lora_num=lora_num,
          per_lora=per_lora,
          dtype="float16",
          device=device,
      )
      outputs = [
        f"h1={in_features}",
        f"h2={weight_size}",
        f"lora_num={lora_num}",
        f"per_lora={per_lora}",
        f"bs={sum(t.request_bins):2d}"
    ]
      layer_idx = 0
      scale = 1.0
      alpha = 1
      beta = 0

      problem_sizes = []
      tensor_As = []
      tensor_Bs = []
      tensor_Cs = []
      tensor_Ds = []
      cur_pos = 0
      # Initialize tensors for each problem in the group
      for idx, req_num in enumerate(t.request_bins):
        problem_sizes.append(cutlass_bindings.gemm.GemmCoord(req_num, weight_size, in_features))
        tensor_As.append(t.x[cur_pos:cur_pos+req_num])
        tensor_Bs.append(t.w_T_deduplicate[idx])
        tensor_Cs.append(t.y[cur_pos:cur_pos+req_num])
        tensor_Ds.append(t.y[cur_pos:cur_pos+req_num])
        cur_pos += req_num
        idx += 1
      
      arguments = GemmGroupedArguments(
          operation, problem_sizes, tensor_As, tensor_Bs, tensor_Cs, tensor_Ds,
          output_op=operation.epilogue_type(alpha, beta)
      )
      # test correctness
      gemm(operation, arguments)
      arguments.sync()
      slora._kernels.dispatch_bgmv(
          t.y_ref, t.x, t.w_T_all.view(-1, 32, 4096//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale)
      assert_close(t.y_ref, t.y)

      result = bench(lambda: gemm(operation, arguments))
      outputs.append(f"\n gemm: {result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")

      result2 = bench(lambda: slora._kernels.dispatch_bgmv(
          t.y_ref, t.x, t.w_T_all.view(-1, 32, 4096//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale))
      outputs.append(f"\n bgmv: {result2.avg()*1e6:3.0f}us±{result2.std()*1e6:3.0f}us")
    except torch.cuda.OutOfMemoryError:
      outputs.append("OOM")

    print(" | ".join(outputs))

class batch_loraA_multi_param:
  def __init__(
      self,
      num_layers: int,
      in_features: int,
      out_features: List[int],
      lora_rank: List[int],
      max_lora_rank: int,
      lora_num: int,
      per_lora: int,
      dtype: str,
      device: torch.device,
  ):
    assert(len(lora_rank) == len(out_features) == lora_num)
    dtype = getattr(torch, dtype)
    self.request_bins = torch.randint(per_lora, per_lora+1, (lora_num,), dtype=torch.long, device=device)
    batch_size = sum(self.request_bins)
    self.x = torch.randn((batch_size, in_features), dtype=dtype, device=device)
    self.y = [torch.zeros((self.request_bins[idx], lora_rank), dtype=dtype, device=device) for idx, lora_rank in enumerate(lora_rank)]
    self.y_ref = torch.zeros((batch_size, max_lora_rank), dtype=dtype, device=device)
    self.y_test = torch.zeros((batch_size, max_lora_rank), dtype=dtype, device=device)
    
    self.w_T_all = torch.randn(
        (16384, 1, 1, in_features),
        dtype=dtype,
        device=device)
    self.w_T_deduplicate = torch.randn(
        (lora_num, num_layers, max_lora_rank, in_features),
        dtype=dtype,
        device=device)
    # self.indicies = torch.arange(0, batch_size, dtype=torch.long, device=device)
    self.indicies = torch.repeat_interleave(torch.arange(0, lora_num, dtype=torch.long, device=device), self.request_bins)
    self.indicies_per_request = torch.arange(0, lora_num, dtype=torch.long, device=device)
    
    values = torch.arange(0, 16384, dtype=torch.long, device=device)
    values = values[torch.randperm(len(values))]
    self.cell_indicies = values[:sum(lora_rank)]
    for idx, lora in enumerate(self.w_T_deduplicate):
        for i in range(out_features[idx]):
            self.w_T_all[self.cell_indicies[sum(lora_rank[:idx])+i],0,0,:].copy_(lora[0,i,:])
    self.lora_ranks = torch.tensor(lora_rank, dtype=torch.long, device=device)
    self.start_indicies = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device=device), self.lora_ranks[:-1]]), dim=0)
    self.lora_ranks = self.lora_ranks * 4
    self.scale = torch.tensor([1.0]*lora_num, dtype=torch.float16, device=device)
    self.b_start_loc = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device=device), self.request_bins[:-1]]), dim=0)

def bench_ggemm_A_multi():
  lora_ranks = [[64]]
  # combo = [(1, 64), (4, 16), (4, 64), (16, 4), (16, 16), (16, 64), (64, 1), (64, 4), (64, 16)]
  combo = [(10, 1)]
  weight_size = 4096
  device = torch.device("cuda:0")

  print("bench_ggemm")
  # for (in_features,
  #      out_features), batch_size in itertools.product(all_sizes, batch_sizes):
  for out_features, (lora_num, per_lora) in itertools.product(lora_ranks, combo):
    lora_random_req = [random.choice(out_features) for _ in range(lora_num)]
    torch.manual_seed(0xabcdabcd987)
    try:
      gc_torch()
      t = batch_loraA_multi_param(
          num_layers=1,
          in_features=weight_size,
          out_features=lora_random_req,
          lora_rank = lora_random_req,
          max_lora_rank = max(lora_random_req),
          lora_num=lora_num,
          per_lora=per_lora,
          dtype="float16",
          device=device,
      )
      outputs = [
        f"h1={weight_size}",
        f"h2={lora_random_req}",
        f"max_lora_rank={max(lora_random_req)}",
        f"lora_num={lora_num}",
        f"per_lora={per_lora}",
        f"bs={sum(t.request_bins):2d}"
    ]
      layer_idx = 0
      scale = 1.0
      alpha = 1
      beta = 0

      problem_sizes = []
      tensor_As = []
      tensor_Bs = []
      tensor_Cs = []
      tensor_Ds = []
      cur_pos = 0
      # Initialize tensors for each problem in the group
      for idx, req_num in enumerate(t.request_bins):
        problem_sizes.append(cutlass_bindings.gemm.GemmCoord(req_num, lora_random_req[idx], weight_size))
        tensor_As.append(t.x[cur_pos:cur_pos+req_num])
        tensor_Bs.append(t.w_T_deduplicate[idx, 0, :lora_random_req[idx], :])
        tensor_Cs.append(t.y[idx])
        tensor_Ds.append(t.y[idx])
        cur_pos += req_num
        idx += 1
      
      arguments = GemmGroupedArguments(
          operation, problem_sizes, tensor_As, tensor_Bs, tensor_Cs, tensor_Ds,
          output_op=operation.epilogue_type(alpha, beta)
      )
      # test correctness
      gemm(operation, arguments)
      arguments.sync()
      padded_tensor = [torch.cat((tensor, torch.zeros(len(tensor), max(lora_random_req) - tensor.shape[1], dtype=tensor.dtype, device=device)), dim=1) for tensor in t.y]
      y_concat = torch.cat(padded_tensor, dim=0)
      assert(len(y_concat) == len(t.y_ref))
      slora._kernels.dispatch_bgmv(
          t.y_ref, t.x, t.w_T_all.view(-1, 32, 4096//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale)
      lora_get_qkvo_fwd_shrink(t.x, t.w_T_all.view(-1, weight_size), t.y_test, t.cell_indicies, 
                               t.start_indicies, t.lora_ranks, t.b_start_loc, 
                               t.request_bins, t.indicies_per_request, weight_size, 0, max(lora_random_req), max(t.request_bins))
      print("max ", torch.max(torch.abs(t.y_test - y_concat)))
      print("mean ", torch.mean(torch.abs(t.y_test - y_concat)))
      print(t.y_test[0,:20])
      print(y_concat[0,:20])
      # assert_close(t.y_ref, y_concat)
      # assert_close(t.y_test, y_concat)

      result = bench(lambda: gemm(operation, arguments))
      outputs.append(f"\n gemm: {result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")

      result2 = bench(lambda: slora._kernels.dispatch_bgmv(
          t.y_ref, t.x, t.w_T_all.view(-1, 32, 4096//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale))
      outputs.append(f"\n bgmv: {result2.avg()*1e6:3.0f}us±{result2.std()*1e6:3.0f}us")

      result3 = bench(lambda: lora_get_qkvo_fwd_shrink(t.x, t.w_T_all.view(-1, weight_size), t.y_test, t.cell_indicies, 
                                                       t.start_indicies, t.lora_ranks, t.b_start_loc, 
                                                       t.request_bins, t.indicies_per_request, weight_size, 
                                                       0, max(lora_random_req), max(t.request_bins)))
      outputs.append(f"\n triton: {result3.avg()*1e6:3.0f}us±{result3.std()*1e6:3.0f}us")
    except torch.cuda.OutOfMemoryError:
      outputs.append("OOM")

    print(" | ".join(outputs))

class batch_loraB_multi_param:
  def __init__(
      self,
      num_layers: int,
      in_features: List[int],
      out_features: int,
      lora_rank: List[int],
      max_lora_rank: int,
      lora_num: int,
      per_lora: int,
      dtype: str,
      device: torch.device,
  ):
    assert(len(lora_rank) == len(in_features) == lora_num)
    dtype = getattr(torch, dtype)
    self.request_bins = torch.randint(per_lora, per_lora+1, (lora_num,), dtype=torch.long, device=device)
    batch_size = sum(self.request_bins)
    # self.x = torch.randn((batch_size, max_lora_rank), dtype=dtype, device=device)
    self.x = [torch.randn((self.request_bins[idx], lora_rank), dtype=dtype, device=device) for idx, lora_rank in enumerate(lora_rank)]
    padded_tensor = [torch.cat((tensor, torch.zeros(len(tensor), max_lora_rank - tensor.shape[1], dtype=tensor.dtype, device=device)), dim=1) for tensor in self.x]
    self.x_cat = torch.cat(padded_tensor, dim=0).contiguous()

    self.y = torch.ones((batch_size, out_features), dtype=dtype, device=device)
    self.y_ref = torch.ones((batch_size, out_features), dtype=dtype, device=device)
    self.y_test = torch.ones((batch_size, out_features), dtype=dtype, device=device)
    
    self.w_T_all = torch.randn(
        (16384, 1, 1, out_features),
        dtype=dtype,
        device=device)
    self.w_T_deduplicate = torch.randn(
        (lora_num, num_layers, max_lora_rank, out_features),
        dtype=dtype,
        device=device)
    # self.indicies = torch.arange(0, batch_size, dtype=torch.long, device=device)
    self.indicies = torch.repeat_interleave(torch.arange(0, lora_num, dtype=torch.long, device=device), self.request_bins)
    self.indicies_per_request = torch.arange(0, lora_num, dtype=torch.long, device=device)

    values = torch.arange(0, 16384, dtype=torch.long, device=device)
    values = values[torch.randperm(len(values))]
    self.cell_indicies = values[:sum(lora_rank)]
    for idx, lora in enumerate(self.w_T_deduplicate):
        for i in range(in_features[idx]):
            self.w_T_all[self.cell_indicies[sum(lora_rank[:idx])+i],0,0,:].copy_(lora.view(max_lora_rank, out_features)[i])
    self.lora_ranks = torch.tensor(lora_rank, dtype=torch.long, device=device)
    self.start_indicies = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device=device), self.lora_ranks[:-1]]), dim=0)
    self.b_start_loc = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device=device), self.request_bins[:-1]]), dim=0)
    self.lora_ranks = self.lora_ranks * 4
    self.scale = torch.tensor([1]*lora_num, dtype=torch.float16, device=device)

def bench_ggemm_B_multi():
  lora_ranks = [[16, 32, 64]]
  # combo = [(1, 64), (4, 16), (4, 64), (16, 4), (16, 16), (16, 64), (64, 1), (64, 4), (64, 16)]
  # combo = [(16,2), (64,1), (16,10), (32,10),(10, 1024)]
  combo = [(200,1)]
  weight_size = 3328
  device = torch.device("cuda:0")

  print("bench_ggemm")
  # for (in_features,
  #      out_features), batch_size in itertools.product(all_sizes, batch_sizes):
  for in_features, (lora_num, per_lora) in itertools.product(lora_ranks, combo):
    lora_random_req = [random.choice(in_features) for _ in range(lora_num)]
    torch.manual_seed(0xabcdabcd987)
    try:
      gc_torch()
      t = batch_loraB_multi_param(
          num_layers=1,
          in_features=lora_random_req,
          out_features=weight_size,
          lora_rank = lora_random_req,
          max_lora_rank = max(lora_random_req),
          lora_num=lora_num,
          per_lora=per_lora,
          dtype="float16",
          device=device,
      )
      outputs = [
        f"h1={lora_random_req}",
        f"h2={weight_size}",
        f"max_lora_rank={max(lora_random_req)}",
        f"lora_num={lora_num}",
        f"per_lora={per_lora}",
        f"bs={sum(t.request_bins):2d}"
    ]
      layer_idx = 0
      scale = 1.0
      alpha = 1.0
      beta = 1

      problem_sizes = []
      tensor_As = []
      tensor_Bs = []
      tensor_Cs = []
      triton_Cs = []
      tensor_Ds = []
      cur_pos = 0
      # Initialize tensors for each problem in the group
      for idx, req_num in enumerate(t.request_bins):
        problem_sizes.append(cutlass_bindings.gemm.GemmCoord(req_num, weight_size, lora_random_req[idx]))
        tensor_As.append(t.x[idx])
        tensor_Bs.append(t.w_T_deduplicate[idx, 0, :lora_random_req[idx], :])
        tensor_Cs.append(t.y[cur_pos:cur_pos+req_num])
        triton_Cs.append(t.y_test[cur_pos:cur_pos+req_num])
        tensor_Ds.append(t.y[cur_pos:cur_pos+req_num])
        cur_pos += req_num
        idx += 1
      
      arguments = GemmGroupedArguments(
          operation, problem_sizes, tensor_As, tensor_Bs, tensor_Cs, tensor_Ds,
          output_op=operation.epilogue_type(alpha, beta)
      )
      # test correctness
      gemm(operation, arguments)
      arguments.sync()
      slora._kernels.dispatch_bgmv(
          t.y_ref, t.x_cat, t.w_T_all.view(-1, 32, weight_size//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale)
      lora_get_qkvo_fwd_expand(t.x_cat, t.w_T_all.view(-1, weight_size), t.y_test, t.scale, t.cell_indicies, 
                               t.start_indicies, t.lora_ranks, t.b_start_loc, 
                               t.request_bins, t.indicies_per_request, weight_size, 
                               0, max(lora_random_req), max(t.request_bins))
      print("max ", torch.max(torch.abs(t.y_test - t.y)))
      print("mean ", torch.mean(torch.abs(t.y_test - t.y)))
      print(t.y_test[0,:20])
      print(t.y[0,:20])
      print(t.y_ref[0,:20])
      assert_close(t.y_ref, t.y)
      assert_close(t.y_test, t.y)
      

      result = bench(lambda: gemm(operation, arguments))
      outputs.append(f"\n gemm: {result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")

      result2 = bench(lambda: slora._kernels.dispatch_bgmv(
          t.y_ref, t.x_cat, t.w_T_all.view(-1, 32, weight_size//32), t.start_indicies, t.lora_ranks, t.cell_indicies, t.indicies, 0, t.scale))
      outputs.append(f"\n bgmv: {result2.avg()*1e6:3.0f}us±{result2.std()*1e6:3.0f}us")

      result3 = bench(lambda: lora_get_qkvo_fwd_expand(t.x_cat, t.w_T_all.view(-1, weight_size), t.y_test, t.scale, t.cell_indicies, 
                                                       t.start_indicies, t.lora_ranks, t.b_start_loc, 
                                                       t.request_bins, t.indicies_per_request, weight_size, 
                                                       0, max(lora_random_req),max(t.request_bins)))
      outputs.append(f"\n triton: {result3.avg()*1e6:3.0f}us±{result3.std()*1e6:3.0f}us")
    except torch.cuda.OutOfMemoryError:
      outputs.append("OOM")

    print(" | ".join(outputs))

BENCH_FN = {
    "ggemmA": bench_ggemm_A,
    "ggemmB": bench_ggemm_B,
    "ggemmA-multi": bench_ggemm_A_multi,
    "ggemmB-multi": bench_ggemm_B_multi,
}


def bench_one_op():
  parser = argparse.ArgumentParser()
  parser.add_argument("--op", choices=BENCH_FN.keys(), required=True)
  args = parser.parse_args()
  bench_fn = BENCH_FN[args.op]
  bench_fn()


if __name__ == "__main__":
  bench_one_op()
