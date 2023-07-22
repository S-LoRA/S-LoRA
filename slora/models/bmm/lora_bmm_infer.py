import numpy as np
import torch
import torch.nn as nn
from typing import final

from slora.common.infer_utils import init_bloc
from slora.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from slora.utils.infer_utils import mark_cost_time


class LoraBmmInfer:

    def __init__(self, base_model, adapters, adapter_sep):

        self.base_model = base_model
        # self.adapters = adapters

        self.adapter_sep = adapter_sep

        # adapter_0, adapter_1, adapter_0
        self.adapter_num = len(adapters)
        self.adapter_scalings = [adapter.scaling if adapter is not None else None for adapter in adapters]

        self.batch_lora_emb = None

        emb_dim = self.base_model.layers_infer[0].embed_dim_ 

        # lora_layer_dim = [adapter.layers[0].k_lora_A.shape[1] if adapter is not None else 0 for adapter in adapters]
        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        max_lora_dim = max(lora_layer_dim)

        # [layer_num, qkvo, adapter_num, hidden_size, lora_dim]
        batch_lora_A = torch.zeros((self.base_model.layers_num, 4, len(adapters), emb_dim, max_lora_dim), dtype=torch.float16, device="cuda")
        batch_lora_B = torch.zeros((self.base_model.layers_num, 4, len(adapters), max_lora_dim, emb_dim), dtype=torch.float16, device="cuda")
        
        # here we can make it better
        for layer_id in range(self.base_model.layers_num):
            for i, adapter in enumerate(adapters):
                if adapter is None: continue
                batch_lora_A[layer_id,0,i,:,:lora_layer_dim[i]].copy_(adapter.layers[layer_id].q_lora_A)
                batch_lora_B[layer_id,0,i,:lora_layer_dim[i],:].copy_(adapter.layers[layer_id].q_lora_B)
                batch_lora_A[layer_id,1,i,:,:lora_layer_dim[i]].copy_(adapter.layers[layer_id].k_lora_A)
                batch_lora_B[layer_id,1,i,:lora_layer_dim[i],:].copy_(adapter.layers[layer_id].k_lora_B)
                batch_lora_A[layer_id,2,i,:,:lora_layer_dim[i]].copy_(adapter.layers[layer_id].v_lora_A)
                batch_lora_B[layer_id,2,i,:lora_layer_dim[i],:].copy_(adapter.layers[layer_id].v_lora_B)
                batch_lora_A[layer_id,3,i,:,:lora_layer_dim[i]].copy_(adapter.layers[layer_id].o_lora_A)
                batch_lora_B[layer_id,3,i,:lora_layer_dim[i],:].copy_(adapter.layers[layer_id].o_lora_B)
        self.batch_lora_A = batch_lora_A
        self.batch_lora_B = batch_lora_B



    def init_batch_lora(self, batch_size, adapter_sep):

        emb_dim = self.base_model.layers_infer[0].embed_dim_ 

        adapter_token_length = []
        for i in range(self.adapter_num-1):
            adapter_token_length.append(adapter_sep[i+1] - adapter_sep[i])
        adapter_token_length.append(batch_size - adapter_sep[-1])
        max_adapter_token_length = max(adapter_token_length)
        
                
        # [adapter_num, hidden_size, emb_dim]
        batch_lora_emb = torch.zeros((self.adapter_num, max_adapter_token_length, emb_dim), dtype=torch.float16, device="cuda")

        self.batch_lora_emb = batch_lora_emb


    # @torch.inference_mode()
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
        assert len(self.adapter_sep) == self.adapter_num
        assert len(b_loc) == len(b_start_loc) == len(b_seq_len)

        self.init_batch_lora(batch_size, self.adapter_sep)

        if is_prefill:
            return self._prefill(batch_size, total_token_num, max_len_in_batch,
                                 input_ids, self.adapter_sep,
                                 b_loc, b_start_loc, b_seq_len)
        else:
            return self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids, self.adapter_sep,
                                b_loc, b_start_loc, b_seq_len)


    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids, adapter_sep,
                 b_loc, b_start_loc, b_seq_len):

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

        infer_state.adapter_sep = adapter_sep
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
        
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics


    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids, adapter_sep,
                b_loc, b_start_loc, b_seq_len):
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.adapter_sep = adapter_sep

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
        predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics


    @final
    def _context_forward(self, input_ids, infer_state):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.context_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_context_forward(i, input_embs, infer_state)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _token_forward(self, input_ids, infer_state):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state):
        self._lora_context_attention(layer_id, input_embs, infer_state)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        layer_infer._context_ffn(input_embs, infer_state, layer_weight)
        return input_embs


    @final
    def _lora_token_forward(self, layer_id, input_embs, infer_state):
        self._lora_token_attention(layer_id, input_embs, infer_state)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        layer_infer._token_ffn(input_embs, infer_state, layer_weight)
        return input_embs


    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _lora_context_attention(self, layer_id, input_embs, infer_state):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._lora_get_o(layer_id, o, infer_state)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # residual
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return


    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        if self.batch_lora_emb is None:
            q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state)
        else:
            q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        if self.batch_lora_emb is None:
            o = self._lora_get_o(layer_id, o, infer_state)
        else:
            o = self._batch_lora_get_o(layer_id, o, infer_state)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return
    

    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # print(f"batch_lora_emb shape: {self.batch_lora_emb.shape}, input_embs shape: {input_embs.shape}")

        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is not None:
                start = infer_state.adapter_sep[i]
                end = infer_state.adapter_sep[i+1] if i+1 < len(infer_state.adapter_sep) else infer_state.batch_size
                self.batch_lora_emb[i,:end-start,:].copy_(input_embs.view(-1, base_layer_infer.embed_dim_)[start:end,:])
        # q (bs, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)

        delta_q = torch.bmm(torch.bmm(self.batch_lora_emb, self.batch_lora_A[layer_id,0]), self.batch_lora_B[layer_id,0]).view(self.adapter_num, -1, base_layer_infer.embed_dim_)

        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is not None:
                start = infer_state.adapter_sep[i]
                end = infer_state.adapter_sep[i+1] if i+1 < len(infer_state.adapter_sep) else infer_state.batch_size
                q[start:end, :] += delta_q[i, :end-start, :] * adapter_scaling

        rotary_emb_fwd(q.view(-1, base_model.tp_k_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        delta_k = torch.bmm(torch.bmm(self.batch_lora_emb, self.batch_lora_A[layer_id,1]), self.batch_lora_B[layer_id,1]).view(self.adapter_num, -1, base_layer_infer.embed_dim_)

        for i, adapter in enumerate(self.adapter_scalings):
            if adapter_scaling is not None:
                start = infer_state.adapter_sep[i]
                end = infer_state.adapter_sep[i+1] if i+1 < len(infer_state.adapter_sep) else infer_state.batch_size
                cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[start:end, :] += delta_k[i, :end-start, :] * adapter_scaling

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        delta_v = torch.bmm(torch.bmm(self.batch_lora_emb, self.batch_lora_A[layer_id,2]), self.batch_lora_B[layer_id,2]).view(self.adapter_num, -1, base_layer_infer.embed_dim_)

        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is not None:
                start = infer_state.adapter_sep[i]
                end = infer_state.adapter_sep[i+1] if i+1 < len(infer_state.adapter_sep) else infer_state.batch_size
                cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_)[start:end, :] += delta_v[i, :end-start, :] * adapter_scaling

        return q        


    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is None: continue
            start = infer_state.b_start_loc[infer_state.adapter_sep[i]]
            if i + 1 == self.adapter_num:
                end = q.shape[0]
            else:
                end = infer_state.b_start_loc[infer_state.adapter_sep[i + 1]]
            input_tensor = input_embs.view(-1, base_layer_infer.embed_dim_)[start: end, :]
            q[start: end, :] += torch.mm(torch.mm(input_tensor,
                                                  self.batch_lora_A[layer_id,0,i]),
                                         self.batch_lora_B[layer_id,0,i]) * adapter_scaling

        rotary_emb_fwd(q.view(-1, base_model.tp_k_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)

        # k (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is None: continue
            start = infer_state.b_start_loc[infer_state.adapter_sep[i]]
            if i + 1 == self.adapter_num:
                end = cache_k.shape[0]
            else:
                end = infer_state.b_start_loc[infer_state.adapter_sep[i + 1]]
            input_tensor = input_embs.view(-1, base_layer_infer.embed_dim_)[start: end, :]
            k_lora = torch.mm(torch.mm(input_tensor,
                                       self.batch_lora_A[layer_id,1,i]),
                              self.batch_lora_B[layer_id,1,i]) * adapter_scaling
            cache_k[start: end, :, :] += k_lora.view(-1, base_model.tp_k_head_num_, base_model.head_dim_)

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is None: continue
            start = infer_state.b_start_loc[infer_state.adapter_sep[i]]
            if i + 1 == self.adapter_num:
                end = cache_v.shape[0]
            else:
                end = infer_state.b_start_loc[infer_state.adapter_sep[i + 1]]
            input_tensor = input_embs.view(-1, base_layer_infer.embed_dim_)[start: end, :]
            v_lora = torch.mm(torch.mm(input_tensor,
                                       self.batch_lora_A[layer_id,2,i]),
                              self.batch_lora_B[layer_id,2,i]) * adapter_scaling
            cache_v[start: end, :, :] += v_lora.view(-1, base_model.tp_k_head_num_, base_model.head_dim_)
        return q
    

    def _batch_lora_get_o(self, layer_id, input, infer_state)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        
        o = torch.mm(input.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                          base_layer_weight.o_weight_)
        
        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is not None:
                start = infer_state.adapter_sep[i]
                end = infer_state.adapter_sep[i+1] if i+1 < len(infer_state.adapter_sep) else infer_state.batch_size
                self.batch_lora_emb[i,:end-start,:].copy_(input.view(-1, base_layer_infer.embed_dim_)[start:end,:])

        delta_o = torch.bmm(torch.bmm(self.batch_lora_emb, self.batch_lora_A[layer_id,3]), self.batch_lora_B[layer_id,3]).view(self.adapter_num, -1, base_layer_infer.embed_dim_)

        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is not None:
                start = infer_state.adapter_sep[i]
                end = infer_state.adapter_sep[i+1] if i+1 < len(infer_state.adapter_sep) else infer_state.batch_size
                o[start:end, :] += delta_o[i, :end-start, :] * adapter_scaling
        
        return o


    def _lora_get_o(self, layer_id, input, infer_state)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_),
                          base_layer_weight.o_weight_)
        for i, adapter_scaling in enumerate(self.adapter_scalings):
            if adapter_scaling is None: continue
            start = infer_state.b_start_loc[infer_state.adapter_sep[i]]
            if i + 1 == self.adapter_num:
                end = o.shape[0]
            else:
                end = infer_state.b_start_loc[infer_state.adapter_sep[i + 1]]
            input_tensor = input.view(-1, base_layer_infer.embed_dim_)[start: end, :]
            o[start: end, :] += torch.mm(torch.mm(input_tensor,
                                                  self.batch_lora_A[layer_id,3,i]),
                                        self.batch_lora_B[layer_id,3,i]) * adapter_scaling
        return o

