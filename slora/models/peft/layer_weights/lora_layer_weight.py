import gc
import torch
import torch.nn as nn


class LoraLayerWeight:
    def __init__(self, layer_num, tp_rank, world_size, lora_config, network_config, data_type=torch.float16,
                 no_lora_swap=False, prefetch_stream=None):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.lora_config = lora_config
        self.network_config = network_config

        # lora params
        self.q_lora_A = None
        self.q_lora_B = None
        self.k_lora_A = None
        self.k_lora_B = None
        self.v_lora_A = None
        self.v_lora_B = None

        self.prefetch_stream = prefetch_stream

        # debug
        self.no_lora_swap = no_lora_swap


    def load_to_torch(self, path):
        numpy_type = {"fp32": np.float32, "fp16": np.float16}[self.data_type_]
        torch_type = {"fp32": torch.float32, "fp16": torch.float16}[self.data_type_]
        return torch.from_numpy(np.fromfile(path, dtype=numpy_type)).to(torch_type)


    def load_dummy_weights(self, swap):
        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        rank = self.lora_config["r"]
        if not swap or self.no_lora_swap:
            self.q_lora_A = (torch.rand((rank, split_n_embed), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.q_lora_B = (torch.rand((split_n_embed, rank), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.k_lora_A = (torch.rand((rank, split_n_embed), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.k_lora_B = (torch.rand((split_n_embed, rank), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.v_lora_A = (torch.rand((rank, split_n_embed), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.v_lora_B = (torch.rand((split_n_embed, rank), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.o_lora_A = (torch.rand((rank, split_n_embed), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
            self.o_lora_B = (torch.rand((split_n_embed, rank), 
                                       dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3
        else:
            self.q_lora_A_home = ((torch.rand((rank, split_n_embed), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.q_lora_A = None
            self.q_lora_B_home = ((torch.rand((split_n_embed, rank), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.q_lora_B = None
            self.k_lora_A_home = ((torch.rand((rank, split_n_embed), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.k_lora_A = None
            self.k_lora_B_home = ((torch.rand((split_n_embed, rank), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.k_lora_B = None
            self.v_lora_A_home = ((torch.rand((rank, split_n_embed), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.v_lora_A = None
            self.v_lora_B_home = ((torch.rand((split_n_embed, rank), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.v_lora_B = None
            self.o_lora_A_home = ((torch.rand((rank, split_n_embed), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.o_lora_A = None
            self.o_lora_B_home = ((torch.rand((split_n_embed, rank), 
                                            dtype=self.data_type_, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3).to("cpu")
            self.o_lora_B = None

            num_head = self.network_config["num_attention_heads"]
            self.w_combined_home = torch.concat(
                [self.q_lora_A_home.T.reshape(rank, num_head, -1),
                 self.k_lora_A_home.T.reshape(rank, num_head, -1),
                 self.v_lora_A_home.T.reshape(rank, num_head, -1),
                 self.o_lora_A_home.T.reshape(rank, num_head, -1),
                 self.q_lora_B_home.T.reshape(rank, num_head, -1),
                 self.k_lora_B_home.T.reshape(rank, num_head, -1),
                 self.v_lora_B_home.T.reshape(rank, num_head, -1),
                 self.o_lora_B_home.T.reshape(rank, num_head, -1)]).pin_memory()
            self.w_combined_home = self.w_combined_home.reshape(2, 4 * rank, num_head, -1)
            self.w_combined = None
        return
 

    def load_hf_weights(self, weights, swap=False, dummy=False):
        if dummy:
            self.load_dummy_weights(swap)
            return

        if swap and not self.no_lora_swap:
            self.load_hf_weights_cpu(weights)
            return

        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_

        prefix = list(weights.keys())[0]
        prefix = prefix[:prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"
        tp_idx = (split_n_embed * self.tp_rank_, split_n_embed * (self.tp_rank_ + 1))

        # q_proj A, B
        if f"{prefix}.q_proj.lora_A.weight" in weights:
            self.q_lora_A = weights[f"{prefix}.q_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.q_lora_A = self.q_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.q_lora_A = self.q_lora_A.cuda()

        if f"{prefix}.q_proj.lora_B.weight" in weights:
            self.q_lora_B = weights[f"{prefix}.q_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.q_lora_B = self.q_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.q_lora_B = self.q_lora_B.cuda()

        # k_proj A, B
        if f"{prefix}.k_proj.lora_A.weight" in weights:
            self.k_lora_A = weights[f"{prefix}.k_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.k_lora_A = self.k_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.k_lora_A = self.k_lora_A.cuda()

        if f"{prefix}.k_proj.lora_B.weight" in weights:
            self.k_lora_B = weights[f"{prefix}.k_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.k_lora_B = self.k_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.k_lora_B = self.k_lora_B.cuda()

        # v_proj A, B
        if f"{prefix}.v_proj.lora_A.weight" in weights:
            self.v_lora_A = weights[f"{prefix}.v_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.v_lora_A = self.v_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.v_lora_A = self.v_lora_A.cuda()

        if f"{prefix}.v_proj.lora_B.weight" in weights:
            self.v_lora_B = weights[f"{prefix}.v_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.v_lora_B = self.v_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.v_lora_B = self.v_lora_B.cuda()

        # o_proj A, B
        if f"{prefix}.o_proj.lora_A.weight" in weights:
            self.o_lora_A = weights[f"{prefix}.o_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.o_lora_A = self.o_lora_A.transpose(0, 1).contiguous().to(self.data_type_)
            self.o_lora_A = self.o_lora_A.cuda()

        if f"{prefix}.o_proj.lora_B.weight" in weights:
            self.o_lora_B = weights[f"{prefix}.o_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.o_lora_B = self.o_lora_B.transpose(0, 1).contiguous().to(self.data_type_)
            self.o_lora_B = self.o_lora_B.cuda()

        return


    def load_hf_weights_cpu(self, weights):
        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_

        prefix = list(weights.keys())[0]
        prefix = prefix[:prefix.find("layers")] + f"layers.{self.layer_num_}.self_attn"
        tp_idx = (split_n_embed * self.tp_rank_, split_n_embed * (self.tp_rank_ + 1))

        # q_proj A, B
        if f"{prefix}.q_proj.lora_A.weight" in weights:
            self.q_lora_A_home = weights[f"{prefix}.q_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.q_lora_A_home = self.q_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.q_lora_A = None

        if f"{prefix}.q_proj.lora_B.weight" in weights:
            self.q_lora_B_home = weights[f"{prefix}.q_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.q_lora_B_home = self.q_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.q_lora_B = None

        # k_proj A, B
        if f"{prefix}.k_proj.lora_A.weight" in weights:
            self.k_lora_A_home = weights[f"{prefix}.k_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.k_lora_A_home = self.k_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.k_lora_A = None

        if f"{prefix}.k_proj.lora_B.weight" in weights:
            self.k_lora_B_home = weights[f"{prefix}.k_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.k_lora_B_home = self.k_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.k_lora_B = None

        # v_proj A, B
        if f"{prefix}.v_proj.lora_A.weight" in weights:
            self.v_lora_A_home = weights[f"{prefix}.v_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.v_lora_A_home = self.v_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.v_lora_A = None

        if f"{prefix}.v_proj.lora_B.weight" in weights:
            self.v_lora_B_home = weights[f"{prefix}.v_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.v_lora_B_home = self.v_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.v_lora_B = None

        # o_proj A, B
        if f"{prefix}.o_proj.lora_A.weight" in weights:
            self.o_lora_A_home = weights[f"{prefix}.o_proj.lora_A.weight"][:, tp_idx[0]:tp_idx[1]]
            self.o_lora_A_home = self.o_lora_A_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.o_lora_A = None

        if f"{prefix}.o_proj.lora_B.weight" in weights:
            self.o_lora_B_home = weights[f"{prefix}.o_proj.lora_B.weight"][tp_idx[0]:tp_idx[1], :]
            self.o_lora_B_home = self.o_lora_B_home.transpose(0, 1).contiguous().to(self.data_type_).pin_memory()
            self.o_lora_B = None
        
        rank = self.lora_config["r"]
        num_head = self.network_config["num_attention_heads"]
        self.w_combined_home = torch.concat(
            [self.q_lora_A_home.T.reshape(rank, num_head, -1),
                self.k_lora_A_home.T.reshape(rank, num_head, -1),
                self.v_lora_A_home.T.reshape(rank, num_head, -1),
                self.o_lora_A_home.T.reshape(rank, num_head, -1),
                self.q_lora_B_home.T.reshape(rank, num_head, -1),
                self.k_lora_B_home.T.reshape(rank, num_head, -1),
                self.v_lora_B_home.T.reshape(rank, num_head, -1),
                self.o_lora_B_home.T.reshape(rank, num_head, -1)]).pin_memory()
        self.w_combined_home = self.w_combined_home.reshape(2, 4 * rank, num_head, -1)
        self.w_combined = None

        return


    def load_to_gpu(self, prefetch=False, bmm=False):
        if not bmm:
            if self.w_combined is None:
                if prefetch:
                    self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)
                else:
                    self.w_combined = self.w_combined_home.to("cuda", non_blocking=True)
        else:
            if self.q_lora_A is None:
                self.q_lora_A = self.q_lora_A_home.to("cuda", non_blocking=True)
                self.q_lora_B = self.q_lora_B_home.to("cuda", non_blocking=True)
                self.k_lora_A = self.k_lora_A_home.to("cuda", non_blocking=True)
                self.k_lora_B = self.k_lora_B_home.to("cuda", non_blocking=True)
                self.v_lora_A = self.v_lora_A_home.to("cuda", non_blocking=True)
                self.v_lora_B = self.v_lora_B_home.to("cuda", non_blocking=True)
                self.o_lora_A = self.o_lora_A_home.to("cuda", non_blocking=True)
                self.o_lora_B = self.o_lora_B_home.to("cuda", non_blocking=True)
 

    def offload_from_gpu(self):
        if self.no_lora_swap:
            return
        #assert self.q_lora_A is not None
        self.w_combined = None
        self.q_lora_A = None
        self.q_lora_B = None
        self.k_lora_A = None
        self.k_lora_B = None
        self.v_lora_A = None
        self.v_lora_B = None
        self.o_lora_A = None
        self.o_lora_B = None
