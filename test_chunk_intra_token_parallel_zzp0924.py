#!/usr/bin/env python3
import os
import time

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import torch_npu  # noqa: F401
import triton

from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_kernel_intra_token_parallel
from fla.ops.kda.chunk_intra_token_parallel_zzp0924 import (
    chunk_kda_fwd_kernel_intra_token_parallel_zzp0924,
)


def launch(kernel, q, k, g, beta, aqk, akk, *, B, T, H, HV, K, BT, BC, BH,
           num_warps=2):
    kernel[(B * T, triton.cdiv(HV, BH))](
        q=q, k=k, g=g, beta=beta, Aqk=aqk, Akk=akk,
        scale=K ** -0.5, cu_seqlens=None, N=B, T=T,
        H=H, HV=HV, K=K, BK=triton.next_power_of_2(K),
        BT=BT, BC=BC, BH=BH, IS_VARLEN=False, USE_GRAPH=False,
        num_warps=num_warps, num_stages=2,
    )


def main():
    torch.manual_seed(20260924)
    B, T, H, HV, K, BT, BC, BH = 1, 64, 32, 32, 128, 64, 16, 8
    q = torch.randn((B, T, H, K), device="npu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    g = torch.randn((B, T, HV, K), device="npu", dtype=torch.float32).cumsum(1) * 0.01
    beta = torch.randn((B, T, HV), device="npu", dtype=torch.bfloat16)

    aqk_ref = torch.zeros((B, T, HV, BT), device="npu", dtype=torch.bfloat16)
    akk_ref = torch.zeros((B, T, HV, BC), device="npu", dtype=torch.float32)
    aqk_opt = torch.zeros_like(aqk_ref)
    akk_opt = torch.zeros_like(akk_ref)

    old = chunk_kda_fwd_kernel_intra_token_parallel.fn.fn
    new = chunk_kda_fwd_kernel_intra_token_parallel_zzp0924.fn.fn
    kwargs = dict(B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BH=BH)
    launch(old, q, k, g, beta, aqk_ref, akk_ref, **kwargs)
    launch(new, q, k, g, beta, aqk_opt, akk_opt, **kwargs)
    torch.npu.synchronize()

    aqk_diff = (aqk_ref.float() - aqk_opt.float()).abs()
    akk_diff = (akk_ref - akk_opt).abs()
    print(f"Aqk bitwise_equal={torch.equal(aqk_ref, aqk_opt)} max_abs={aqk_diff.max().item()}")
    print(f"Akk bitwise_equal={torch.equal(akk_ref, akk_opt)} max_abs={akk_diff.max().item()}")
    torch.testing.assert_close(aqk_opt, aqk_ref, rtol=0, atol=0)
    torch.testing.assert_close(akk_opt, akk_ref, rtol=0, atol=0)
    print("CORRECTNESS_OK")

    def bench(label, kernel, bh, num_warps, warmup=20, rep=100):
        args = dict(B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BH=bh)
        for _ in range(warmup):
            launch(kernel, q, k, g, beta, aqk_opt, akk_opt,
                   num_warps=num_warps, **args)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(rep):
            launch(kernel, q, k, g, beta, aqk_opt, akk_opt,
                   num_warps=num_warps, **args)
        torch.npu.synchronize()
        us = (time.perf_counter() - start) * 1e6 / rep
        print(f"BENCH {label}: {us:.3f} us")
        return us

    for trial in range(5):
        bench(f"trial{trial}_old_BH8_W2", old, 8, 2, warmup=30, rep=300)
        bench(f"trial{trial}_new_BH8_W2", new, 8, 2, warmup=30, rep=300)


if __name__ == "__main__":
    main()
