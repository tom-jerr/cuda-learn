"""Small boundary suite for compute-sanitizer (run after bench.py builds)."""
import ctypes
import sys
from pathlib import Path

import torch
import tvm_ffi

folder = Path(sys.argv[1]).resolve()
handles = [ctypes.CDLL(str(folder / f"cute_s{s}.so")) for s in (2, 3)]
for stage in (2, 3):
    fn = tvm_ffi.get_global_func(f"fa2_case.forward_s{stage}")
    for n in (64, 128, 192, 320):
        q, k, v = [torch.randn(1, 1, n, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        out = torch.empty_like(q)
        for causal in (False, True):
            with tvm_ffi.use_torch_stream():
                fn(q, k, v, out, int(causal))
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
print("Sanitizer boundary suite completed")
