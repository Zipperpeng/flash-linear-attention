"""One-program-per-token KDA prototype with a vectorized BJ=4 dimension."""

import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.jit
def chunk_kda_fwd_kernel_intra_token_parallel_bj4vec_zzp0924(
    q, k, g, beta, Aqk, Akk, scale,
    T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BH: tl.constexpr, BJ: tl.constexpr,
):
    i_tg = tl.program_id(0).to(tl.int32)
    i_hg = tl.program_id(1).to(tl.int32)
    bos = (i_tg // T) * T
    i_t = i_tg % T
    i_c = i_t // BT
    i_s = (i_t % BT) // BC
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

    tqk = i_t.to(tl.int64) * H * K
    tg = i_t.to(tl.int64) * HV * K
    b_q = tl.load(q + tqk + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_k = tl.load(k + tqk + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_g = tl.load(g + tg + p_hvk, mask=m_hk, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + i_t.to(tl.int64) * HV + o_hv,
                     mask=m_hv, other=0.0).to(tl.float32)
    b_k *= b_beta[:, None]

    loop_count = min(i_t + 1, min(T, i_ts + BC)) - i_ts
    for jb in range(0, loop_count, BJ):
        o_j = jb + tl.arange(0, BJ)
        j = i_ts + o_j
        m_j = o_j < loop_count
        p_kj = k + j[:, None, None].to(tl.int64) * H * K + p_qk[None, :, :]
        p_gj = g + j[:, None, None].to(tl.int64) * HV * K + p_hvk[None, :, :]
        mask_jhk = m_j[:, None, None] & m_hk[None, :, :]
        b_kj = tl.load(p_kj, mask=mask_jhk, other=0).to(tl.float32)
        b_gj = tl.load(p_gj, mask=mask_jhk, other=0.0).to(tl.float32)
        b_kgj = tl.where(
            mask_jhk,
            b_kj * exp2(b_g[None, :, :] - b_gj),
            0.0,
        )
        b_Aqk = tl.sum(b_q[None, :, :] * b_kgj, axis=2) * scale
        b_Akk = tl.sum(b_k[None, :, :] * b_kgj, axis=2)
        b_Akk *= tl.where(j[:, None] < i_t, 1.0, 0.0)

        p_aqk = (Aqk + i_t.to(tl.int64) * HV * BT
                 + o_hv[None, :] * BT + i_s * BC + o_j[:, None])
        p_akk = (Akk + i_t.to(tl.int64) * HV * BC
                 + o_hv[None, :] * BC + o_j[:, None])
        mask_jh = m_j[:, None] & m_hv[None, :]
        tl.store(p_aqk, b_Aqk.to(Aqk.dtype.element_ty), mask=mask_jh)
        tl.store(p_akk, b_Akk.to(Akk.dtype.element_ty), mask=mask_jh)
