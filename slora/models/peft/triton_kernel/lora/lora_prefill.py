import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

if triton.__version__ >= "2.1.0":
    @triton.jit
    def _expand_fwd_kernel(
        X, W, scale, B_Loc, B_Lora_Start_Loc, B_Lora_Ranks, B_Start_Loc, B_Seqlen, B_Indicies,
        Out,
        qkvo,
        stride_xbs, stride_xh,
        stride_wbs, stride_wh,
        stride_obs, stride_oh,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_RANK: tl.constexpr,
        TILE_N: tl.constexpr
    ):
        cur_batch = tl.program_id(0)
        cur_tile = tl.program_id(1)
        start_m = tl.program_id(2)
        cur_adapter = tl.load(B_Indicies + cur_batch)

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_rank_size = tl.load(B_Lora_Ranks + cur_adapter) // 4
        cur_batch_adapter_start_index = tl.load(B_Lora_Start_Loc + cur_adapter) + cur_batch_rank_size * qkvo
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_scale = tl.load(scale + cur_adapter)

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_RANK)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_x = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_xbs + offs_d[None, :] * stride_xh
        x = tl.load(X + off_x, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

        for start_n in range(cur_tile * TILE_N, (cur_tile+1)*TILE_N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute xw ----
            w_loc = tl.load(B_Loc + cur_batch_adapter_start_index + ((start_n + offs_n)*cur_batch_rank_size//BLOCK_DMODEL), mask=(start_n + offs_n) < BLOCK_DMODEL, other=0)
            off_w = w_loc[None, :] * stride_wbs + (((start_n + offs_n)*cur_batch_rank_size+offs_d[:, None])%BLOCK_DMODEL) * stride_wh
            w = tl.load(W + off_w, mask=offs_d[:, None] < cur_batch_rank_size, other=0.0)
            
            off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + (start_n + offs_n[None, :]) * stride_oh
            out_ptrs = Out + off_o
            wx = tl.load(out_ptrs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

            wx += tl.dot(x, w) * cur_batch_scale

            tl.store(out_ptrs, wx, mask=offs_m[:, None] < cur_batch_seq_len)

        return
    
    @triton.jit
    def _shrink_fwd_kernel(
        X, W, B_Loc, B_Lora_Start_Loc, B_Lora_Ranks, B_Start_Loc, B_Seqlen, B_Indicies,
        Out,
        qkvo,
        stride_xbs, stride_xh,
        stride_wbs, stride_wh,
        stride_obs, stride_oh,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        start_n = tl.program_id(1)
        start_m = tl.program_id(2)
        cur_adapter = tl.load(B_Indicies + cur_batch)

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_rank_size = tl.load(B_Lora_Ranks + cur_adapter) // 4
        cur_batch_adapter_start_index = tl.load(B_Lora_Start_Loc + cur_adapter) + cur_batch_rank_size * qkvo
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_x = (cur_batch_in_all_start_index + offs_m) * stride_xbs

        offs_k = tl.arange(0, BLOCK_K)
        
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        w_loc = tl.load(B_Loc + cur_batch_adapter_start_index + offs_n, mask=offs_n < cur_batch_rank_size, other=0)
        off_w = w_loc * stride_wbs
        
        wx = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        for start_k in range(0, BLOCK_DMODEL, BLOCK_K):
            start_k = tl.multiple_of(start_k, BLOCK_K)
            # -- compute xw ----
            x = tl.load(X + off_x[:, None] + (start_k+offs_k[None, :]) * stride_xh, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
            w = tl.load(W + off_w[None, :] + (start_k+offs_k[:, None]) * stride_wh, mask=offs_n[None, :] < cur_batch_rank_size, other=0.0)
            wx += tl.dot(x, w)
        
        c = wx.to(tl.float16)
        # initialize pointers to output
        off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + offs_n[None, :] * stride_oh
        out_ptrs = Out + off_o
        tl.store(out_ptrs, c, mask=offs_m[:, None] < cur_batch_seq_len)

        return

    @torch.inference_mode()
    def lora_get_qkvo_fwd_expand(x, w, o, scale, b_loc, b_lora_start, b_lora_ranks, b_start_loc, b_seq_len, b_indicies, feat_out, qkvo, max_rank, max_input_len):
        # good for large input_len (prefill stage) better than bgmv, worse than cutlass
        BLOCK_N = 128
        N = 1
        TILE = N * BLOCK_N
        BLOCK_M = 32
        # BLOCK_N = 16
        # N = 32
        # TILE = N * BLOCK_N
        # BLOCK_M = 16

        batch = b_seq_len.shape[0]

        grid = (batch, triton.cdiv(feat_out, TILE), triton.cdiv(max_input_len, BLOCK_M))  # batch, head,

        num_warps = 4
        _expand_fwd_kernel[grid](
            x, w, scale, b_loc, b_lora_start, b_lora_ranks, b_start_loc, b_seq_len, b_indicies,
            o,
            qkvo,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            o.stride(0), o.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=feat_out,
            BLOCK_N=BLOCK_N,
            BLOCK_RANK=max_rank,
            TILE_N=TILE,
            num_warps=num_warps,
            num_stages=2,
        )
        return
    
    @torch.inference_mode()
    def lora_get_qkvo_fwd_shrink(x, w, o, b_loc, b_lora_start, b_lora_ranks, b_start_loc, b_seq_len, b_indicies, hidden_size, qkvo, max_rank, max_input_len):
        # good for large input_len (prefill stage) better than bgmv, worse than cutlass
        BLOCK_N = 16 if max_rank > 8 else max_rank
        BLOCK_M = 32
        BLOCK_K = 128

        batch = b_seq_len.shape[0]

        grid = (batch, triton.cdiv(max_rank, BLOCK_N), triton.cdiv(max_input_len, BLOCK_M))  # batch, head,

        num_warps = 4
        _shrink_fwd_kernel[grid](
            x, w, b_loc, b_lora_start, b_lora_ranks, b_start_loc, b_seq_len, b_indicies,
            o,
            qkvo,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            o.stride(0), o.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=hidden_size,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=1,
        )
        return