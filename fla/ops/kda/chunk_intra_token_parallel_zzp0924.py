# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental KDA intra-token kernel optimized on 2026-09-24.

The original implementation is intentionally kept unchanged. This safe
candidate specializes fixed sequence lengths and hoists segment pointer
arithmetic out of the inner loop. It remains a separate A/B-test entry point.
"""

import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H", "HV"],
    **autotune_cache_kwargs,
)
@triton.jit
def chunk_kda_fwd_kernel_intra_token_parallel_zzp0924(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    # Keep control indices in i32. Convert only memory offsets that may exceed
    # the i32 range to i64 below.
    i_tg = tl.program_id(0).to(tl.int32)
    i_hg = tl.program_id(1).to(tl.int32)

    if IS_VARLEN:
        if USE_GRAPH and i_tg >= tl.load(cu_seqlens + N).to(tl.int32):
            return
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1

        i_n = left
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        sequence_length = eos - bos
        i_t = i_tg - bos
    else:
        # T is specialized in this variant, so fixed-length division/modulo
        # and their dependent expressions can be folded by the compiler.
        bos = (i_tg // T) * T
        i_t = i_tg % T
        sequence_length = T

    if i_t >= sequence_length:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_ts = i_c * BT + i_s * BC

    G: tl.constexpr = HV // H

    bos64 = bos.to(tl.int64)
    q += bos64 * H * K
    k += bos64 * H * K
    g += bos64 * HV * K
    Aqk += bos64 * HV * BT
    Akk += bos64 * HV * BC
    beta += bos64 * HV

    o_hv = i_hg * BH + tl.arange(0, BH)
    o_h = o_hv // G
    o_k = tl.arange(0, BK)
    m_hv = o_hv < HV
    m_k = o_k < K
    m_hk = m_hv[:, None] & m_k[None, :]

    p_qk = o_h[:, None] * K + o_k[None, :]
    token_qk_offset = i_t.to(tl.int64) * H * K
    b_q = tl.load(q + token_qk_offset + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_k = tl.load(k + token_qk_offset + p_qk, mask=m_hk, other=0).to(tl.float32)

    p_hvk = o_hv[:, None] * K + o_k[None, :]
    token_g_offset = i_t.to(tl.int64) * HV * K
    b_g = tl.load(g + token_g_offset + p_hvk, mask=m_hk, other=0.0).to(tl.float32)
    p_beta = beta + i_t.to(tl.int64) * HV + o_hv
    b_k *= tl.load(p_beta, mask=m_hv, other=0.0).to(tl.float32)[:, None]

    # Hoist all invariant segment/output bases.  dj is used directly by both
    # stores, avoiding j % BT and j - i_ts in every iteration.
    segment_k = k + i_ts.to(tl.int64) * H * K + p_qk
    segment_g = g + i_ts.to(tl.int64) * HV * K + p_hvk
    out_aqk = Aqk + i_t.to(tl.int64) * HV * BT + o_hv * BT + i_s * BC
    out_akk = Akk + i_t.to(tl.int64) * HV * BC + o_hv * BC
    loop_count = min(i_t + 1, min(sequence_length, i_ts + BC)) - i_ts

    for dj in range(0, loop_count):
        j = i_ts + dj
        b_kj = tl.load(segment_k + dj * H * K, mask=m_hk, other=0).to(tl.float32)
        b_gj = tl.load(segment_g + dj * HV * K, mask=m_hk, other=0.0).to(tl.float32)

        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(out_aqk + dj, b_Aqk.to(Aqk.dtype.element_ty), mask=m_hv)
        tl.store(out_akk + dj, b_Akk.to(Akk.dtype.element_ty), mask=m_hv)


def chunk_kda_fwd_intra_token_parallel_zzp0924(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    use_graph: bool = False,
) -> None:
    """Run the independent zzp0924 optimization candidate in-place."""
    B, T, H, K, HV = *q.shape, gk.shape[2]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size
    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (B * T, triton.cdiv(HV, meta['BH']))

    chunk_kda_fwd_kernel_intra_token_parallel_zzp0924[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BK=BK,
        BT=BT,
        BC=BC,
        USE_GRAPH=use_graph,
    )
    return Aqk, Akk
