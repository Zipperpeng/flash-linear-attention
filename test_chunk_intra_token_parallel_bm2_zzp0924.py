#!/usr/bin/env python3
import os
import time

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import torch_npu  # noqa: F401
import triton

from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_kernel_intra_token_parallel
from fla.ops.kda.chunk_intra_token_parallel_bm2_zzp0924 import (
    chunk_kda_fwd_kernel_intra_token_parallel_bm2_zzp0924,
)


def main():
    torch.manual_seed(20260924)
    B, T, H, HV, K, BT, BC = 1, 64, 32, 32, 128, 64, 16
    q = torch.randn((B, T, H, K), device="npu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    g = torch.randn((B, T, HV, K), device="npu", dtype=torch.float32).cumsum(1) * 0.01
    beta = torch.randn((B, T, HV), device="npu", dtype=torch.bfloat16)
    aqk_ref = torch.zeros((B, T, HV, BT), device="npu", dtype=torch.bfloat16)
    akk_ref = torch.zeros((B, T, HV, BC), device="npu", dtype=torch.float32)
    aqk_opt, akk_opt = torch.zeros_like(aqk_ref), torch.zeros_like(akk_ref)
    old = chunk_kda_fwd_kernel_intra_token_parallel.fn.fn
    new = chunk_kda_fwd_kernel_intra_token_parallel_bm2_zzp0924

    def run_old():
        old[(B * T, triton.cdiv(HV, 8))](
            q, k, g, beta, aqk_ref, akk_ref, K ** -0.5, None, B, T,
            H=H, HV=HV, K=K, BK=K, BT=BT, BC=BC, BH=8,
            IS_VARLEN=False, USE_GRAPH=False, num_warps=2, num_stages=2)

    def run_new(bh, warps=2):
        new[(B * T // 2, triton.cdiv(HV, bh))](
            q, k, g, beta, aqk_opt, akk_opt, K ** -0.5,
            T=T, H=H, HV=HV, K=K, BK=K, BT=BT, BC=BC, BH=bh,
            num_warps=warps, num_stages=2)

    run_old()
    run_new(8)
    torch.npu.synchronize()
    print("Aqk", torch.equal(aqk_ref, aqk_opt), (aqk_ref.float() - aqk_opt.float()).abs().max().item())
    print("Akk", torch.equal(akk_ref, akk_opt), (akk_ref - akk_opt).abs().max().item())
    torch.testing.assert_close(aqk_opt, aqk_ref, rtol=0, atol=0)
    torch.testing.assert_close(akk_opt, akk_ref, rtol=0, atol=0)
    print("CORRECTNESS_OK")

    def bench(label, fn, warmup=30, rep=300):
        for _ in range(warmup): fn()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(rep): fn()
        torch.npu.synchronize()
        value = (time.perf_counter() - start) * 1e6 / rep
        print(f"{label}: {value:.3f} us")

    for trial in range(3):
        bench(f"trial{trial}_old_BH8", run_old)
        bench(f"trial{trial}_bm2_BH8_W1", lambda: run_new(8, 1))
        bench(f"trial{trial}_bm2_BH8_W2", lambda: run_new(8, 2))
        bench(f"trial{trial}_bm2_BH8_W4", lambda: run_new(8, 4))
        bench(f"trial{trial}_bm2_BH16_W2", lambda: run_new(16, 2))


if __name__ == "__main__":
    main()
