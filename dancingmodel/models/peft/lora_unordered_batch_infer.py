import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import final

from dancingmodel.common.infer_utils import init_bloc
from dancingmodel.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from dancingmodel.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from dancingmodel.models.llama.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from dancingmodel.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from dancingmodel.utils.infer_utils import mark_cost_time
from dancingmodel.utils.infer_utils import calculate_time, mark_start, mark_end
from dancingmodel._kernels import dispatch_bgmv


class LoraUnorderedBatchInfer:

    def __init__(self, base_model, adapters, infer_adapter=None):
        self.base_model = base_model

        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        self.max_lora_dim = max(lora_layer_dim)

        self.req_bins = torch.zeros(len(adapters), dtype=torch.long, device="cuda")

        if infer_adapter is not None:
            self.infer_adapter = infer_adapter
            if isinstance(infer_adapter, NaiveInferAdapter):
                self.key_buffer = infer_adapter.key_buffer
                self.value_buffer = infer_adapter.value_buffer
            else:
                self.key_buffer = infer_adapter.mem_manager.key_buffer
                self.value_buffer = infer_adapter.mem_manager.value_buffer
            for i, adapter in enumerate(adapters):
                # FIX ME @TODO: currently not supporting adapter is None
                if adapter is None: continue
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx

        self.world_size = self.base_model.world_size_
        self.tp_rank = self.base_model.tp_rank_

    @torch.no_grad()
    def forward(
            self,
            batch_size, # number of request
            total_token_num,
            max_len_in_batch,
            input_ids, # 1D input tensor
            b_loc, # mapping to memory pool
            b_start_loc, # the start index of each request
            b_seq_len, # the current length of each request
            is_prefill=True,
            use_bmm=True,
            no_lora_compute=False,
            no_lora_copy=False):

        # Notice that batch_lora only support decoding
        assert len(b_loc) == len(b_start_loc) == len(b_seq_len)

        if self.world_size > 1:
            self.strip_start = self.infer_adapter.a_start // self.world_size
            self.strip_len = self.infer_adapter.a_len // self.world_size
            self.strip_loc = self.infer_adapter.a_loc[::self.world_size].contiguous() // self.world_size
        else:
            self.strip_start = self.infer_adapter.a_start
            self.strip_len = self.infer_adapter.a_len
            self.strip_loc = self.infer_adapter.a_loc

        self.delta = torch.empty((len(input_ids), self.max_lora_dim), dtype=torch.float16, device="cuda")
        self.deltalist = []

        if self.world_size > 1:
            self.delta_tp_part = torch.empty((len(input_ids), self.max_lora_dim // self.world_size), dtype=torch.float16, device="cuda")
            self.delta_o_tp_part = torch.empty((len(input_ids), self.base_model.embed_dim_ // self.world_size), dtype=torch.float16, device="cuda")
            self.delta_all_gather_buf = torch.empty((self.world_size, len(input_ids), self.max_lora_dim // self.world_size), dtype=torch.float16, device="cuda")
        else:
            self.delta_tp_part = self.delta

        self.max_b_seq_len = torch.max(b_seq_len).item()

        if is_prefill:
            assert(len(self.req_bins)==len(b_seq_len))
            self.batch_req_bins = torch.repeat_interleave(self.req_bins, b_seq_len)
            # self.b_start_loc = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device="cuda"), b_seq_len[:-1]]), dim=0)
            for _ in range(3):
                self.deltalist.append(torch.zeros((len(self.batch_req_bins), self.max_lora_dim), dtype=torch.float16, device="cuda"))

            self.rank_sizes = self.strip_len[self.batch_req_bins] // 4
            return self._prefill(batch_size, total_token_num, max_len_in_batch,
                                 input_ids,
                                 b_loc, b_start_loc, b_seq_len, no_lora_compute)
        else:
            self.rank_sizes = self.strip_len[self.req_bins] // 4
            return self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids,
                                b_loc, b_start_loc, b_seq_len,
                                no_lora_compute, no_lora_copy)


    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_start_loc, b_seq_len, no_lora_compute=False):

        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        b_seq_len_numpy = b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                        for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
                self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
                self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.mem_manager = self.base_model.mem_manager
        infer_state.prefill_mem_index = self.base_model.mem_manager.alloc(infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)

        predict_logics = self._context_forward(input_ids, infer_state, no_lora_compute)
        return predict_logics


    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids,
                b_loc, b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False):
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        infer_state.mem_manager = self.base_model.mem_manager

        alloc_mem = self.base_model.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.base_model.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        infer_state.init_some_extra_state(self.base_model, batch_size, total_token_num, max_len_in_batch,
                                          input_ids, b_loc, b_start_loc, b_seq_len, False)
        predict_logics = self._token_forward(input_ids, infer_state, no_lora_compute, no_lora_copy)
        return predict_logics


    @final
    def _context_forward(self, input_ids, infer_state, no_lora_compute=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.context_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_context_forward(i, input_embs, infer_state, no_lora_compute)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _token_forward(self, input_ids, infer_state, no_lora_compute=False, no_lora_copy=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state, no_lora_compute, no_lora_copy)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        self._lora_context_attention(layer_id, input_embs, infer_state, no_lora_compute)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        layer_infer._context_ffn(input_embs, infer_state, layer_weight)
        return input_embs


    @final
    # @calculate_time(show=True, min_cost_ms=0)
    def _lora_token_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        self._lora_token_attention(layer_id, input_embs, infer_state, no_lora_compute, no_lora_copy)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # mark_start("token_ffn")
        layer_infer._token_ffn(input_embs, infer_state, layer_weight)
        # mark_end("token_ffn")
        return input_embs


    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _lora_context_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)
        if self.world_size > 1: #all reduce after o
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # residual
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return


    # @calculate_time(show=True, min_cost_ms=0)
    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._batch_lora_get_o(layer_id, o, infer_state, no_lora_compute)
        if self.world_size > 1: #all reduce after o
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return


    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False, no_lora_copy=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # q (bs, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)

        # @TODO: fix me, filter requests querying only base model

        if not no_lora_compute:
            # Part A of lora
            dispatch_bgmv(self.delta_tp_part, input_embs.view(-1, base_layer_infer.embed_dim_),
                          self.key_buffer[layer_id],
                          self.strip_start, self.strip_len,
                          self.strip_loc, self.req_bins, 0, self.infer_adapter.a_scaling)

            if self.world_size > 1:
               # need all gather -> (s, r/m) -> (s, r)
               all_gather_token(self.delta, self.delta_tp_part, self.delta_all_gather_buf,
                                self.rank_sizes)

            # Part B of lora
            dispatch_bgmv(q, self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.req_bins, 0, self.infer_adapter.a_scaling)

        rotary_emb_fwd(q.view(-1, base_model.tp_k_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # Part A of lora
            dispatch_bgmv(self.delta_tp_part, input_embs.view(-1, base_layer_infer.embed_dim_),
                          self.key_buffer[layer_id],
                          self.strip_start, self.strip_len,
                          self.strip_loc, self.req_bins, 1, self.infer_adapter.a_scaling)

            # need all gather -> (bs, r/m) -> (bs, r)
            if self.world_size > 1:
                # (1, sum of r)
                all_gather_token(self.delta, self.delta_tp_part, self.delta_all_gather_buf,
                                 self.rank_sizes)

            # Part B of lora
            dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                          self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.req_bins, 1, self.infer_adapter.a_scaling)

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # Part A of lora
            dispatch_bgmv(self.delta_tp_part, input_embs.view(-1, base_layer_infer.embed_dim_),
                          self.key_buffer[layer_id],
                          self.strip_start, self.strip_len,
                          self.strip_loc, self.req_bins, 2, self.infer_adapter.a_scaling)

            # need all gather -> (bs, r/m) -> (bs, r)
            if self.world_size > 1:
                # (1, sum of r)
                all_gather_token(self.delta, self.delta_tp_part, self.delta_all_gather_buf,
                                 self.rank_sizes)

            # Part B of lora
            dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                          self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                          self.infer_adapter.a_len, self.infer_adapter.a_loc,
                          self.req_bins, 2, self.infer_adapter.a_scaling)

        return q


    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)

        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2 and self.world_size == 1:
            # if 1 == 0:
                # TODO this branch needs to be re-tested
                delta_qA = self.deltalist[0]
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         delta_qA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_qA, self.value_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         q, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(self.delta_tp_part, input_embs.view(-1, base_layer_infer.embed_dim_),
                              self.key_buffer[layer_id],
                              self.strip_start, self.strip_len,
                              self.strip_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling)
                if self.world_size > 1:
                    # need all gather -> (s, r/m) -> (s, r)
                    all_gather_token(self.delta, self.delta_tp_part, self.delta_all_gather_buf,
                                     self.rank_sizes)
                dispatch_bgmv(q, self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                            self.infer_adapter.a_len, self.infer_adapter.a_loc,
                            self.batch_req_bins, 0, self.infer_adapter.a_scaling)

        rotary_emb_fwd(q.view(-1, base_model.tp_k_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)
 
        # k (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            delta_kA = self.deltalist[1]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2 and self.world_size == 1:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         delta_kA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_kA, self.value_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(self.delta_tp_part, input_embs.view(-1, base_layer_infer.embed_dim_),
                            self.key_buffer[layer_id],
                            self.strip_start, self.strip_len,
                            self.strip_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling)

                if self.world_size > 1:
                    # need all gather -> (s, r/m) -> (s, r)
                    all_gather_token(self.delta, self.delta_tp_part, self.delta_all_gather_buf,
                                     self.rank_sizes)

                dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                            self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                            self.infer_adapter.a_len, self.infer_adapter.a_loc,
                            self.batch_req_bins, 1, self.infer_adapter.a_scaling)

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if not no_lora_compute:
            delta_vA = self.deltalist[2]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2 and self.world_size == 1:
            # if 1 ==0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         delta_vA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_vA, self.value_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(self.delta_tp_part, input_embs.view(-1, base_layer_infer.embed_dim_),
                              self.key_buffer[layer_id],
                              self.strip_start, self.strip_len,
                              self.strip_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling)

                if self.world_size > 1:
                    # need all gather -> (s, r/m) -> (s, r)
                    all_gather_token(self.delta, self.delta_tp_part, self.delta_all_gather_buf,
                                     self.rank_sizes)

                dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                              self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.batch_req_bins, 2, self.infer_adapter.a_scaling)
        return q


    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                          base_layer_weight.o_weight_)

        if not no_lora_compute:
            dispatch_bgmv(self.delta, input.view(-1, base_layer_infer.embed_dim_ // self.world_size),
                          self.key_buffer[layer_id],
                          self.infer_adapter.a_start, self.infer_adapter.a_len,
                          self.infer_adapter.a_loc, self.req_bins, 3, self.infer_adapter.a_scaling)

            # need to all reduce before applying lora B
            if self.world_size > 1:
                dist.all_reduce(self.delta, op=dist.ReduceOp.SUM, async_op=False)
                self.delta_o_tp_part.zero_()
                dispatch_bgmv(self.delta_o_tp_part,
                              self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.req_bins, 3, self.infer_adapter.a_scaling)
                # fuse all_gather into the later all reduce
                launch_segmented_add(self.delta_o_tp_part, self.tp_rank, self.world_size, o)
            else:
                dispatch_bgmv(o,
                              self.delta, self.value_buffer[layer_id], self.infer_adapter.a_start,
                              self.infer_adapter.a_len, self.infer_adapter.a_loc,
                              self.req_bins, 3, self.infer_adapter.a_scaling)
        return o


    def _lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                          base_layer_weight.o_weight_)

        if not no_lora_compute:
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2 and self.world_size == 1:
            # if 1 == 0:
                delta_oA = self.deltalist[0]
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         delta_oA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, self.value_buffer[layer_id].view(-1, base_layer_infer.embed_dim_), 
                                         o, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(self.delta, input.view(-1, base_layer_infer.embed_dim_ // self.world_size),
                            self.key_buffer[layer_id],
                            self.infer_adapter.a_start, self.infer_adapter.a_len,
                            self.infer_adapter.a_loc, self.batch_req_bins, 3, self.infer_adapter.a_scaling)

                # need to all reduce before applying lora B
                if self.world_size > 1:
                    dist.all_reduce(self.delta, op=dist.ReduceOp.SUM, async_op=False)
                    self.delta_o_tp_part.zero_()
                    dispatch_bgmv(self.delta_o_tp_part, self.delta,
                                  self.value_buffer[layer_id], self.infer_adapter.a_start,
                                  self.infer_adapter.a_len, self.infer_adapter.a_loc,
                                  self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                    # fuse all_gather into the later all reduce
                    launch_segmented_add(self.delta_o_tp_part, self.tp_rank, self.world_size, o)
        return o


def all_gather_token(out_tensor, in_tensor, inter_tensor, r_size):
    dist.all_gather([inter_tensor[i] for i in range(inter_tensor.shape[0])],
                    in_tensor, group=None, async_op=False)
    launch_segmented_copy(inter_tensor, r_size, out_tensor)


import cupy

segmented_copy_kernel = cupy.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void kernel(const half* __restrict__ all_gather_res,
            const long int* __restrict__ r_size,
            half* __restrict__ output,
            int tp_size, int bs, int partitioned_r) {
  int cur_b = blockIdx.x;
  int cur_r = r_size[cur_b];
  int t_id = threadIdx.x;

  if (t_id >= cur_r * tp_size) {
    return;
  }

  int i = t_id / cur_r;
  int j = t_id % cur_r;

  output[cur_b * partitioned_r * tp_size + t_id] = (
    all_gather_res[i * bs * partitioned_r + cur_b * partitioned_r + j]);
}
''', "kernel")

def launch_segmented_copy(all_gather_res, r_size, output):
    tp_size, bs, partitioned_r = all_gather_res.shape
    assert output.shape[0] == bs and output.shape[1] == tp_size * partitioned_r
    grid_size = (bs,)
    block_size = (tp_size * partitioned_r,)

    segmented_copy_kernel(grid_size, block_size,
        (cupy.asarray(all_gather_res),
         cupy.asarray(r_size),
         cupy.asarray(output),
         tp_size, bs, partitioned_r))


segmented_add_kernel = cupy.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void kernel(const half* __restrict__ in_tensor,
            half* __restrict__ out_tensor,
            int rank, int tp_size, int split_h, int BLOCK_SIZE) {
  int cur_b = blockIdx.x;
  int t_id = threadIdx.x;

  for (int i = t_id; i < split_h; i += BLOCK_SIZE) {
    out_tensor[cur_b * split_h * tp_size + split_h * rank + i] += (
       in_tensor[cur_b * split_h + i]);
  }
}
''', "kernel")


def launch_segmented_add(in_tensor, rank, tp_size, out_tensor):
    BLOCK_SIZE = 512
    grid_size = (len(in_tensor),)
    block_size = (BLOCK_SIZE,)

    segmented_add_kernel(grid_size, block_size,
        (cupy.asarray(in_tensor),
         cupy.asarray(out_tensor),
         rank, tp_size, in_tensor.shape[1],
         BLOCK_SIZE))


if __name__ == "__main__":
    a = torch.ones((10, 10), dtype=torch.float16, device="cuda")
    b = torch.ones((10, 10), dtype=torch.float16, device="cuda")
    launch_segmented_add(a, 0, 0, b)
