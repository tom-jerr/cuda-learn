#!/usr/bin/env python3
"""CPU algebra/address checks for ce088ab FA4 explanation, float64 not BF16 simulation."""
import numpy as np


def close(label, x, y):
    err=float(np.max(np.abs(x-y)))
    np.testing.assert_allclose(x,y,rtol=2e-11,atol=2e-11)
    print(f'{label}: PASS (max abs {err:.3g})')


def main():
    rng=np.random.default_rng(4)
    q=rng.normal(size=(512,128));k=rng.normal(size=(384,128));v=rng.normal(size=(384,128))
    z=(q@k.T)/np.sqrt(128)/np.log(2)
    # Force both skipped and taken rescale branches in reverse block traversal.
    z[:,128:256]+=3.0;z[:,:128]+=14.0
    p=np.exp2(z-z.max(axis=1,keepdims=True));ref=p@v/p.sum(axis=1,keepdims=True)
    out=np.empty_like(q)
    taken=skipped=0
    for stage in range(2):
        qs=slice(stage*256,(stage+1)*256)
        b=None;l=np.zeros(256);o=np.zeros((256,128))
        for j in (2,1,0):
            ns=slice(j*128,(j+1)*128)
            zz=z[qs,ns];vj=v[ns]
            if b is None:
                b=zz.max(axis=1);alpha=np.zeros(256)
            else:
                candidate=np.maximum(b,zz.max(axis=1));change=candidate-b>8
                taken+=int(change.sum());skipped+=int((~change).sum())
                bn=np.where(change,candidate,b);alpha=np.exp2(b-bn);b=bn
            pj=np.exp2(zz-b[:,None])
            part=pj[:,:96]@vj[:96]+pj[:,96:]@vj[96:]
            close(f'PV split stage{stage} block{j}',part,pj@vj)
            o=alpha[:,None]*o+part;l=alpha*l+pj.sum(axis=1)
        out[qs]=o/l[:,None]
    assert taken and skipped
    close('conditional online softmax; two Q stages',out,ref)
    print(f'rescale branch rows: taken={taken}, skipped={skipped}')

    # Two-CTA A/C rows, B columns; concatenate logical subproducts, no reduction split.
    qq=q[:256];kk=k[:128];vv=v[:128]
    ss=np.concatenate([np.concatenate([a@kk[:64].T,a@kk[64:].T],axis=1) for a in (qq[:128],qq[128:])],axis=0)
    close('2CTA QK ownership',ss,qq@kk.T)
    pp=np.exp(ss/np.sqrt(128)-np.max(ss/np.sqrt(128),axis=1,keepdims=True))
    oo=np.concatenate([np.concatenate([a@vv[:,:64],a@vv[:,64:]],axis=1) for a in (pp[:128],pp[128:])],axis=0)
    close('2CTA PV ownership',oo,pp@vv)

    q1=q[:128];k1=k[:256];v1=v[:256];do=rng.normal(size=(128,128));a=1/np.sqrt(128)
    logits=a*q1@k1.T;prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
    o=prob@v1;delta=(do*o).sum(axis=1)
    ds=prob*(do@v1.T-delta[:,None])
    ds_t=prob.T*(v1@do.T-delta[None,:])
    close('backward transposed dS',ds_t,ds.T)
    A=ds[:64,:128];B=ds[:64,128:];C=ds[64:,:128];D=ds[64:,128:]
    dqa=np.concatenate([A,B],axis=1)@k1
    dqb=np.concatenate([C,D],axis=1)@k1
    close('DSM half exchange dQ',a*np.concatenate([dqa,dqb]),a*ds@k1)
    close('2CTA dK KV-row ownership',a*np.concatenate([ds_t[:128]@q1,ds_t[128:]@q1]),a*ds.T@q1)
    close('2CTA dV KV-row ownership',np.concatenate([prob.T[:128]@do,prob.T[128:]@do]),prob.T@do)
    assert A.size*2 == 16384
    assert 2*64*128*4 == (2*128*128*4)//2
    print('DSM 16 KiB per direction and dQ output bytes halved: PASS')

    # Shared physical permutations; element offsets before S<3,4,3> byte swizzle.
    def sw(e):return e ^ ((e & 0x1c0)>>3)
    for name,rows in [('Q',128),('K',64)]:
        offsets=[]
        for r in range(rows):
            for f in range(128):
                e=64*r+(f%64)+(f//64)*rows*64
                enested=r*64+f%16+((f//16)%4)*16+(f//64)*rows*64
                assert e==enested
                assert sw(e)*2 == (2*e)^(((2*e)&0x380)>>3)
                offsets.append(sw(e))
        assert len(set(offsets))==rows*128 and min(offsets)==0 and max(offsets)==rows*128-1
        print(f'{name} nested layout and SW128 bijection: PASS')
    offsets=[]
    for token in range(128):
        for feature in range(64):
            e=feature+(token%16)*64+(token//16)*1024
            assert e==feature+token*64
            offsets.append(sw(e))
    assert len(set(offsets))==8192
    print('V MN-major nested layout and SW128 bijection: PASS')
    print('Scope: float64 CPU algebra + address mappings; no GPU execution/latency/BF16-bitwise claim.')

if __name__=='__main__':main()
