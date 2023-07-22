import time
import torch
import triton
import triton.language as tl

from slora._kernels import dispatch_bgmv


@triton.jit
def triton_batch_lora_B(
    output,
    x,
    w,
    a_start,
    a_len,
    a_loc,
    batch_req_bins,
    a_scaling,
    qkvo_offset: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_LORA_RANK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    return


def batch_lora_forward_B(
    output,
    x,
    w,
    a_start,
    a_len,
    a_loc,
    batch_req_bins,
    qkvo_offset,
    a_scaling,
):
    #print("B", output.shape, x.shape, w.shape, a_start.shape, a_len.shape, a_loc.shape,
    #      batch_req_bins.shape, qkvo_offset, a_scaling.shape)
    NUM_TOKENS, MAX_LORA_RANK = x.shape
    NUM_TOKENS, HIDDEN = output.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(NUM_TOKENS, BLOCK_SIZE_M), triton.cdiv(HIDDEN, BLOCK_SIZE_N))
    triton_batch_lora_B[grid](output, x,
                              w,
                              a_start, a_len, 
                              a_loc, batch_req_bins, a_scaling, qkvo_offset,
                              NUM_TOKENS, HIDDEN, MAX_LORA_RANK,
                              BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)


def test_bgmv():
    H = 1024
    R = [8, 16]
    max_r = max(R)
    num_adapters = 2
    N = 128
    num_head = 32
    part = "B"

    if part == "A":
        x = torch.randn((N, H), dtype=torch.float16, device="cuda")
        delta_qA = torch.zeros((len(x), max_r), dtype=torch.float16, device="cuda")
        forward_func = dispatch_bgmv
    else:
        x = torch.randn((N, max_r), dtype=torch.float16, device="cuda")
        delta_qA = torch.zeros((len(x), H), dtype=torch.float16, device="cuda")
        forward_func = dispatch_bgmv

    key_buffer = torch.concat([
        torch.randn((R[0] * 4, num_head, H // num_head), dtype=torch.float16, device="cuda"),
        torch.randn((R[1] * 4, num_head, H // num_head), dtype=torch.float16, device="cuda"),
    ])
    a_len = torch.tensor([
        R[0] * 4, R[1] * 4
    ], dtype=torch.long, device="cuda")
    a_start = torch.zeros_like(a_len)
    a_start[1:] = torch.cumsum(a_len[:-1], dim=0)
    a_loc = torch.arange((R[0] + R[1]) * 4, dtype=torch.long, device="cuda")
    a_scaling = torch.tensor([1, 1], dtype=torch.float16, device="cuda")
    batch_req_bins = torch.concat([
        torch.tensor([i] * ((N  + num_adapters - 1) // num_adapters),
            dtype=torch.long, device="cuda")
        for i in range(num_adapters)])
    batch_req_bins = batch_req_bins[:len(x)]

    #x[0][R[0]:] = 0
    #x[1][R[1]:] = 0

    qkvo = 0
    results = []
    for i in range(N):
        a_id = batch_req_bins[i]
        a_w = key_buffer[a_start[a_id] + qkvo * R[a_id]: a_start[a_id] + (qkvo + 1) * R[a_id]]
        if part == "A":
            a_w = a_w.reshape(R[a_id], H).T
        else:
            a_w = a_w.reshape(H, R[a_id]).T
        ret = x[i:i+1, :R[a_id]] @ a_w
        results.append(ret)
    ref = delta_qA + torch.concat(results)

    forward_func(delta_qA, x,
                 key_buffer,
                 a_start, a_len,
                 a_loc, batch_req_bins, qkvo, a_scaling)

    #print("delta_qA", delta_qA[0, :10].tolist())
    #print("ref", ref[0, :10].tolist())

    print("max delta 0:", torch.max(torch.abs(delta_qA - ref)))
    #print("max delta 1:", torch.max(torch.abs(delta_qA[1] - ref[1])))

    def to_test():
        #batch_lora_forward_B(delta_qA, x,
        #                     key_buffer,
        #                     a_start, a_len, 
        #                     a_loc, batch_req_bins, 0, a_scaling)

        dispatch_bgmv(delta_qA, x,
                      key_buffer,
                      a_start, a_len, 
                      a_loc, batch_req_bins, 0, a_scaling)
        #ref = x @ key_buffer[:R].reshape(-1, H).T

    # Warm up
    for _ in range(10):
        to_test()
    run_iter = 500
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        to_test()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Time cost {((t2 - t1) / run_iter) * 1000:.2f} ms")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_bgmv()
