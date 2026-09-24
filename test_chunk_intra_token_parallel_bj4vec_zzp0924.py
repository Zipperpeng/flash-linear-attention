#!/usr/bin/env python3
import os, time
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
import torch
import torch_npu  # noqa
import triton
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_kernel_intra_token_parallel
from fla.ops.kda.chunk_intra_token_parallel_bj4vec_zzp0924 import chunk_kda_fwd_kernel_intra_token_parallel_bj4vec_zzp0924

B,T,H,HV,K,BT,BC=1,64,32,32,128,64,16
torch.manual_seed(20260924)
q=torch.randn((B,T,H,K),device="npu",dtype=torch.bfloat16); k=torch.randn_like(q)
g=torch.randn((B,T,HV,K),device="npu",dtype=torch.float32).cumsum(1)*0.01
beta=torch.randn((B,T,HV),device="npu",dtype=torch.bfloat16)
ar=torch.zeros((B,T,HV,BT),device="npu",dtype=torch.bfloat16); kr=torch.zeros((B,T,HV,BC),device="npu")
ao=torch.zeros_like(ar); ko=torch.zeros_like(kr)
old=chunk_kda_fwd_kernel_intra_token_parallel.fn.fn; new=chunk_kda_fwd_kernel_intra_token_parallel_bj4vec_zzp0924
def ro(): old[(B*T,triton.cdiv(HV,8))](q,k,g,beta,ar,kr,K**-.5,None,B,T,H=H,HV=HV,K=K,BK=K,BT=BT,BC=BC,BH=8,IS_VARLEN=False,USE_GRAPH=False,num_warps=2,num_stages=2)
def rn(bh): new[(B*T,triton.cdiv(HV,bh))](q,k,g,beta,ao,ko,K**-.5,T=T,H=H,HV=HV,K=K,BK=K,BT=BT,BC=BC,BH=bh,BJ=4,num_warps=2,num_stages=2)
ro(); rn(8); torch.npu.synchronize()
print("Aqk",torch.equal(ar,ao),(ar.float()-ao.float()).abs().max().item()); print("Akk",torch.equal(kr,ko),(kr-ko).abs().max().item())
torch.testing.assert_close(ao,ar,rtol=0,atol=0); torch.testing.assert_close(ko,kr,rtol=0,atol=0); print("CORRECTNESS_OK")
def bench(n,f):
  for _ in range(30): f()
  torch.npu.synchronize(); s=time.perf_counter()
  for _ in range(300): f()
  torch.npu.synchronize(); print(n,(time.perf_counter()-s)*1e6/300,"us")
for i in range(3): bench(f"old{i}",ro); bench(f"bj4_bh4_{i}",lambda:rn(4)); bench(f"bj4_bh8_{i}",lambda:rn(8))
