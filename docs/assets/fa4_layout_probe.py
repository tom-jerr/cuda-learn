import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as h
from cutlass.cute.nvgpu import tcgen05 as t
@cute.jit
def probe():
    qk = h.make_trivial_tiled_mma(cutlass.BFloat16,t.OperandMajorMode.K,t.OperandMajorMode.K,cutlass.Float32,t.CtaGroup.TWO,(256,128))
    pv = h.make_trivial_tiled_mma(cutlass.BFloat16,t.OperandMajorMode.K,t.OperandMajorMode.MN,cutlass.Float32,t.CtaGroup.TWO,(256,128),t.OperandSource.TMEM)
    print('Q:',h.make_smem_layout_a(qk,(256,128,128),cutlass.BFloat16,2))
    print('K:',h.make_smem_layout_b(qk,(256,128,128),cutlass.BFloat16,6))
    print('V:',h.make_smem_layout_b(pv,(256,128,128),cutlass.BFloat16,6))
    print('Q partition:', qk.partition_shape_A((256,128)))
    print('K partition:', qk.partition_shape_B((128,128)))
    print('V partition:', pv.partition_shape_B((128,128)))
cute.compile(probe)
