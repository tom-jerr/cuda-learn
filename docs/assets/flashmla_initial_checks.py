#!/usr/bin/env python3
"""Executable CPU examples for the accompanying FlashMLA first-version article.

These verify mathematical claims, not Hopper kernel execution or performance.
"""
import numpy as np


def scheduler_example():
    blocks = [2, 3, 40, 7]
    num_parts, overhead = 4, 5
    payload = (sum(n + overhead for n in blocks) + num_parts - 1) // num_parts + overhead
    req, start = 0, 0
    parts = []
    counts = [0] * len(blocks)
    for _ in range(num_parts):
        remain = payload
        part = []
        while req < len(blocks):
            rest = blocks[req] - start
            if remain >= rest + overhead:
                part.append((req, start, blocks[req]))
                counts[req] += 1
                remain -= rest + overhead
                req, start = req + 1, 0
            else:
                if remain > overhead:
                    end = start + remain - overhead
                    part.append((req, start, end))
                    counts[req] += 1
                    start = end
                break
        parts.append(part)
    assert req == 4 and start == 0
    assert payload == 23
    assert parts == [[(0, 0, 2), (1, 0, 3), (2, 0, 3)],
                     [(2, 3, 21)], [(2, 21, 39)], [(2, 39, 40), (3, 0, 7)]]
    assert [0, *np.cumsum(counts)] == [0, 1, 2, 6, 7]
    print('PASS: original scheduler example; costs=23,23,23,18; split prefix=0,1,2,6,7')


def online(x, values, order):
    m = np.full(x.shape[0], -np.inf)
    l = np.zeros(x.shape[0])
    out = np.zeros((x.shape[0], values.shape[1]))
    for tile in order:
        xs = x[:, tile*64:(tile+1)*64]
        vs = values[tile*64:(tile+1)*64]
        new_m = np.maximum(m, xs.max(axis=1))
        safe_m = np.where(np.isneginf(new_m), 0, new_m)
        rho = np.exp(m-safe_m)
        p = np.exp(xs-safe_m[:, None])
        out = rho[:, None]*out+p@vs
        l = rho*l+p.sum(axis=1)
        m = new_m
    normalized = out / np.where(l > 0, l, 1)[:, None]
    lse = np.full_like(l, -np.inf)
    valid = l > 0
    lse[valid] = m[valid] + np.log(l[valid])
    return normalized, lse


def attention_example():
    rng = np.random.default_rng(20250221)
    q = rng.normal(size=(64, 576))
    cache = rng.normal(size=(130, 576))
    pad = np.full((192, 576), np.nan)
    pad[:130] = cache
    border = 126 + np.arange(64)//16
    visible = np.arange(192)[None, :] <= border[:, None]
    scores = q@pad.T / np.sqrt(576)
    # Assignment/selection removes NaN columns; adding -inf would not do so.
    scores = np.where(visible, scores, -np.inf)
    assert not np.isnan(scores).any()
    p = np.exp(scores-scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    unsafe = p@pad[:, :512]
    assert np.isnan(unsafe).all()
    clean_v = np.nan_to_num(pad[:, :512], nan=0.0)
    reference = p@clean_v
    forward, _ = online(scores, clean_v, [0, 1, 2])
    reverse, _ = online(scores, clean_v, [2, 1, 0])
    # The last tile is completely masked for the first 32 query rows.
    tail, tail_lse = online(scores, clean_v, [2])
    assert np.all(tail[:32] == 0) and np.all(np.isneginf(tail_lse[:32]))
    pieces = [online(scores, clean_v, [i]) for i in range(3)]
    partial_o = np.stack([s[0] for s in pieces])
    lses = np.stack([s[1] for s in pieces])
    w = np.exp(lses-lses.max(axis=0))
    w /= w.sum(axis=0)
    combined = np.sum(w[:, :, None]*partial_o, axis=0)
    for actual in [forward, reverse, combined]:
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    print('PASS: causal masks; all-masked tail; forward/reverse online softmax; split-LSE combine')
    print('PASS: masked QK is clean, but zero probabilities times NaN V contaminate PV; clearing tail fixes it')


if __name__ == '__main__':
    scheduler_example()
    attention_example()
