#!/usr/bin/env python3
"""Numerical checks for the tutorial's algebra, not GPU kernel validation.

Uses float64 to isolate scheduling/normalization identities from BF16 rounding.
Checks changing maxima, masked/odd blocks, split+sink, and dual-GEMM reduction.
"""
import numpy as np


def reference(x, v, mask, sink=None):
    x = np.where(mask, x, -np.inf)
    m = x.max()
    p = np.exp(x-m)
    denom = p.sum() + (0 if sink is None else np.exp(sink-m))
    return p@v/denom, m+np.log(p.sum())


def seesaw(x, v, mask, block=64):
    n, d = v.shape
    mid = d//2
    left, right = np.zeros(mid), np.zeros(d-mid)
    m, l0, l1 = -1e30, 0., 0.
    for start in range(0, n, 2*block):
        sl0 = slice(start, min(start+block, n))
        sl1 = slice(min(start+block, n), min(start+2*block, n))
        x0 = np.where(mask[sl0], x[sl0], -np.inf)
        x1 = np.where(mask[sl1], x[sl1], -np.inf)
        m0 = max(m, x0.max(initial=-np.inf))
        a0 = np.exp(m-m0)
        p0 = np.exp(x0-m0)
        left = a0*left + p0@v[sl0,:mid]
        l0 = a0*l0 + p0.sum()
        m1 = max(m0, x1.max(initial=-np.inf))
        a1 = np.exp(m0-m1)
        p1 = np.exp(x1-m1)
        right = a0*a1*right + p1@v[sl1,mid:]
        l1 = a0*a1*l1 + p1.sum()
        left = a1*left + p1@v[sl1,:mid]
        l0 *= a1
        right += (a1*p0)@v[sl0,mid:]
        m = m1
    return np.r_[left,right]/(l0+l1), m+np.log(l0+l1)


def delayed_scale(x, v, block=64):
    # SM100 conceptual one-row version: base need not equal exact running max.
    base, l = -1e30, 0.
    o = np.zeros(v.shape[1])
    for i in range(0,len(x),block):
        xx=x[i:i+block]
        new=max(base,xx.max()) if xx.max()-base>6 else base
        a=np.exp2(base-new)
        p=np.exp2(xx-new)
        o=a*o+p@v[i:i+block]
        l=a*l+p.sum()
        base=new
    return o/l


def main():
    rng=np.random.default_rng(152025)
    cases=0
    for n in [1,63,64,65,127,128,129,193,384,513]:
        for trial in range(8):
            x=rng.normal(size=n)*4+np.arange(n)//64*7
            v=rng.normal(size=(n,32))
            mask=rng.random(n)>.2
            mask[0]=True
            if trial==0 and n>64: mask[64:128]=False
            expected,lse=reference(x,v,mask)
            actual,ls=seesaw(x,v,mask)
            np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=2e-13)
            np.testing.assert_allclose(ls,lse,atol=2e-13)
            cases+=1
    print(f'PASS: seesaw vs full softmax, {cases} cases; odd/tail/masked blocks and changing maxima.')
    x=rng.normal(size=257)*3
    v=rng.normal(size=(257,16))
    chunks=np.array_split(np.arange(257),7)
    local=[reference(x[c],v[c],np.ones(len(c),bool)) for c in chunks]
    sink=5.25
    ls=np.array([r[1] for r in local]); m=ls.max()
    w=np.exp(ls-m)
    out=sum(wi*r[0] for wi,r in zip(w,local))/(w.sum()+np.exp(sink-m))
    np.testing.assert_allclose(out,reference(x,v,np.ones(len(x),bool),sink)[0],atol=2e-14)
    print('PASS: split outputs plus sink once equal direct attention with sink.')
    q=rng.normal(size=(13,512)); k=rng.normal(size=(64,512))
    cols=np.arange(512).reshape(8,64)
    even=cols[::2].ravel(); odd=cols[1::2].ravel()
    dual=q[:,even]@k[:,even].T + q[:,odd]@k[:,odd].T
    np.testing.assert_allclose(dual,q@k.T,atol=1e-12)
    perm=np.r_[cols[4:].ravel(),cols[:4].ravel()]
    np.testing.assert_allclose(q[:,perm]@k[:,perm].T,q@k.T,atol=1e-12)
    print('PASS: dual-GEMM partial score sum and feature-stripe reordered QK.')
    x=rng.normal(size=512)+np.arange(512)//64*1.5
    v=rng.normal(size=(512,16))
    np.testing.assert_allclose(delayed_scale(x,v),reference(x*np.log(2),v,np.ones(512,bool))[0],atol=2e-14)
    assert np.isnan(0*np.nan)
    print('PASS: delayed max rescaling identity; 0 * NaN remains NaN.')


if __name__=='__main__': main()
