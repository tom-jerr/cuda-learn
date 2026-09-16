#!/usr/bin/env python3
"""Check output-column/head partitioning and the address moves in the figures.
Pure CPU algebra checks, not a CUDA kernel test or race detector.
"""
import numpy as np
from flashmla_latest_checks import seesaw, reference

def main():
    rng=np.random.default_rng(64)
    q=rng.normal(size=(128,576))*.1
    kv=rng.normal(size=(193,576))
    logits=q@kv.T
    p=np.exp(logits-logits.max(axis=1,keepdims=True))
    l=p.sum(axis=1,keepdims=True)
    whole=(p@kv[:,:512])/l
    halves=np.concatenate(((p@kv[:,:256])/l,(p@kv[:,256:512])/l),axis=1)
    np.testing.assert_allclose(halves,whole,rtol=1e-12,atol=1e-12)
    # Cluster gathers each selected KV record once then replicates it into both CTAs.
    # The head tiles perform separate softmax, both over every selected token.
    tiles=[]
    for heads in [slice(0,64),slice(64,128)]:
        qq=q[heads];s=qq@kv.T
        pp=np.exp(s-s.max(axis=1,keepdims=True))
        tiles.append(pp@kv[:,:512]/pp.sum(axis=1,keepdims=True))
    np.testing.assert_allclose(np.concatenate(tiles),whole,rtol=1e-12,atol=1e-12)
    # Seesaw also works when two local weight blocks begin with different max bases.
    for h in [0,63,64,127]:
        actual,_=seesaw(logits[h],kv[:,:512],np.ones(193,dtype=bool))
        np.testing.assert_allclose(actual,whole[h],rtol=1e-12,atol=1e-12)
    def sw(r,f):return (f//64)*4096+r*64+((f%64)^(8*(r%8)))
    def inter(r,f):return 8*r+512*(f//8)+f%8
    for r in range(48):
        for f in range(448):
            assert sw(r+16,f)-sw(r,f)==1024
            assert sw(r,f+64)-sw(r,f)==4096
            assert inter(r,f+8)-inter(r,f)==512
    assert sw(1,0)==72 and sw(1,8)==64
    out,_=reference(np.log([2.,3.]),np.array([[10.,100.],[20.,200.]]),np.ones(2,dtype=bool))
    np.testing.assert_allclose(out,[16,160])
    print('PASS: column halves concatenate; cluster head halves concatenate; all heads see full KV.')
    print('PASS: seesaw with shared normalization; toy output [16,160]; figure address strides.')

if __name__=='__main__': main()
