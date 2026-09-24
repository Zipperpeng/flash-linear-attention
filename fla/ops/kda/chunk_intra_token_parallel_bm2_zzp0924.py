"""BM=2 prototype for fixed-length KDA intra-token computation."""

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.jit
def chunk_kda_fwd_kernel_intra_token_parallel_bm2_zzp0924(
    q, k, g, beta, Aqk, Akk, scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
):
    i_pairg = tl.program_id(0).to(tl.int32)
    i_hg = tl.program_id(1).to(tl.int32)
    i_tg0 = i_pairg * 2
    bos = (i_tg0 // T) * T
    i_t0 = i_tg0 % T
    i_t1 = i_t0 + 1

    i_c = i_t0 // BT
    i_s = (i_t0 % BT) // BC
    i_ts = i_c * BT + i_s * BC

    G: tl.constexpr = HV // H
    bos64 = bos.to(tl.int64)
    q += bos64 * H * K
    k += bos64 * H * K
    g += bos64 * HV * K
    beta += bos64 * HV
    Aqk += bos64 * HV * BT
    Akk += bos64 * HV * BC

    o_hv = i_hg * BH + tl.arange(0, BH)
    o_h = o_hv // G
    o_k = tl.arange(0, BK)
    m_hv = o_hv < HV
    m_k = o_k < K
    m_hk = m_hv[:, None] & m_k[None, :]

    p_qk = o_h[:, None] * K + o_k[None, :]
    p_hvk = o_hv[:, None] * K + o_k[None, :]
    t0_qk = i_t0.to(tl.int64) * H * K
    t1_qk = i_t1.to(tl.int64) * H * K
    t0_g = i_t0.to(tl.int64) * HV * K
    t1_g = i_t1.to(tl.int64) * HV * K

    b_q0 = tl.load(q + t0_qk + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_q1 = tl.load(q + t1_qk + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_k0 = tl.load(k + t0_qk + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_k1 = tl.load(k + t1_qk + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_g0 = tl.load(g + t0_g + p_hvk, mask=m_hk, other=0.0).to(tl.float32)
    b_g1 = tl.load(g + t1_g + p_hvk, mask=m_hk, other=0.0).to(tl.float32)
    b_beta0 = tl.load(beta + i_t0.to(tl.int64) * HV + o_hv,
                      mask=m_hv, other=0.0).to(tl.float32)
    b_beta1 = tl.load(beta + i_t1.to(tl.int64) * HV + o_hv,
                      mask=m_hv, other=0.0).to(tl.float32)
    b_k0 *= b_beta0[:, None]
    b_k1 *= b_beta1[:, None]

    segment_k = k + i_ts.to(tl.int64) * H * K + p_qk
    segment_g = g + i_ts.to(tl.int64) * HV * K + p_hvk
    out0_aqk = Aqk + i_t0.to(tl.int64) * HV * BT + o_hv * BT + i_s * BC
    out1_aqk = Aqk + i_t1.to(tl.int64) * HV * BT + o_hv * BT + i_s * BC
    out0_akk = Akk + i_t0.to(tl.int64) * HV * BC + o_hv * BC
    out1_akk = Akk + i_t1.to(tl.int64) * HV * BC + o_hv * BC
    loop_count = min(i_t1 + 1, min(T, i_ts + BC)) - i_ts

    for dj in range(0, loop_count):
        j = i_ts + dj
        b_kj = tl.load(segment_k + dj * H * K, mask=m_hk, other=0).to(tl.float32)
        b_gj = tl.load(segment_g + dj * HV * K, mask=m_hk, other=0.0).to(tl.float32)

        if j <= i_t0:
            b_kgj0 = tl.where(m_k[None, :], b_kj * exp2(b_g0 - b_gj), 0.0)
            b_Aqk0 = tl.sum(b_q0 * b_kgj0, axis=1) * scale
            b_Akk0 = tl.sum(b_k0 * b_kgj0, axis=1) * tl.where(j < i_t0, 1.0, 0.0)
            tl.store(out0_aqk + dj, b_Aqk0.to(Aqk.dtype.element_ty), mask=m_hv)
            tl.store(out0_akk + dj, b_Akk0.to(Akk.dtype.element_ty), mask=m_hv)

        b_kgj1 = tl.where(m_k[None, :], b_kj * exp2(b_g1 - b_gj), 0.0)
        b_Aqk1 = tl.sum(b_q1 * b_kgj1, axis=1) * scale
        b_Akk1 = tl.sum(b_k1 * b_kgj1, axis=1) * tl.where(j < i_t1, 1.0, 0.0)
        tl.store(out1_aqk + dj, b_Aqk1.to(Aqk.dtype.element_ty), mask=m_hv)
        tl.store(out1_akk + dj, b_Akk1.to(Akk.dtype.element_ty), mask=m_hv)


def chunk_kda_fwd_intra_token_parallel_bm2_zzp0924(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    block_heads: int = 8,
    num_warps: int = 2,
):
    B, T, H, K, HV = *q.shape, gk.shape[2]
    if T % 2 != 0:
        raise ValueError("BM=2 prototype requires an even fixed sequence length")
    BT, BC = chunk_size, sub_chunk_size
    grid = (B * T // 2, triton.cdiv(HV, block_heads))
    chunk_kda_fwd_kernel_intra_token_parallel_bm2_zzp0924[grid](
        q, k, gk, beta, Aqk, Akk, scale,
        T=T, H=H, HV=HV, K=K, BK=triton.next_power_of_2(K),
        BT=BT, BC=BC, BH=block_heads,
        num_warps=num_warps, num_stages=2,
    )
    return Aqk, Akk
