#!/usr/bin/env python3
"""Source of truth for the V3.2 SM90 decode teaching diagrams (not timings).

Optional positional argument selects an output directory. Layout equations are
checked by examples/flash_attn/flashmla_address_probe.cu against real CuTe types.
"""
import argparse
from pathlib import Path
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD, GOLD_LIGHT, PANEL, MUTED
import cute_r2s_retile as renderer


def box(d, x, y, w, h, title, lines, fill=PANEL):
    d.rect(x, y, w, h, fill)
    d.text(x+16, y+29, title, 21, bold=True)
    for i, line in enumerate(lines):
        d.text(x+16, y+59+i*28, line, 18)


def arrow(d, x, y, xx, yy):
    assert x == xx or y == yy
    d.parts.append(f'<path d="M{x},{y} L{xx},{yy}" stroke="#172033" stroke-width="2" fill="none"/>')
    if y == yy:
        s = 1 if xx > x else -1
        points = f'{xx},{yy} {xx-8*s},{yy-5} {xx-8*s},{yy+5}'
    else:
        s = 1 if yy > y else -1
        points = f'{xx},{yy} {xx-5},{yy-8*s} {xx+5},{yy-8*s}'
    d.parts.append(f'<polygon points="{points}" fill="#172033"/>')


def ownership():
    d = Drawing('Block / warp / thread：三种不同的切分轴',
                '固定：V3.2，128 query heads，1个 KV head；一个 cluster 的两个 CTA 协作处理同一 query。', 1060)
    for c, x in enumerate((40, 625)):
        d.rect(x, 112, 535, 313, 'white')
        d.text(x+16, 144, f'CTA {c}：heads {c*64}..{c*64+63}，384 threads', 22, bold=True)
        box(d, x+16, 164, 503, 73, 'WG0 · threads 0..127 · warps 0..3',
            ['QK + online softmax + O[:, 0:256]'], BLUE_LIGHT)
        box(d, x+16, 248, 503, 73, 'WG1 · threads 128..255 · warps 4..7',
            ['接收 P / rescale，计算 O[:, 256:512]'], GOLD_LIGHT)
        box(d, x+16, 332, 503, 73, 'WG2 · threads 256..383 · warps 8..11',
            [f'gather / dequant selected rows {c*32}..{c*32+31}'])
    d.text(40, 466, '两个 CTA 用 DSM 互送已解量化的半块；每个 CTA 最终都有完整 K[64,576]。', 21, bold=True)
    d.text(40, 515, 'Producer：表中每列的4个lane处理同一个token，每个lane搬16维。', 22, BLUE, True)
    d.table(40, 535, [220, 225, 225, 225, 225], [
        ['warp内 lane', '0 / 8 / 16 / 24', '1 / 9 / 17 / 25', '2 / 10 / 18 / 26', '… 7 / 15 / 23 / 31'],
        ['selected row', '8w + 0', '8w + 1', '8w + 2', '8w + 7'],
        ['4个lane的feature', '0:16, 16:32,', '每组相同分工', '每组相同分工', '每组相同分工'],
        ['一轮搬64维', '32:48, 48:64', '沿feature重复8轮', '得到512维latent', '另搬64维RoPE'],
    ], rh=42)
    d.text(40, 748, 'Consumer：4个相邻lane共同拥有两行score；每线程持有离散的列。', 22, GOLD, True)
    d.table(40, 770, [220, 225, 225, 225, 225], [
        ['WG0 / warp0', 'lane 0', 'lane 1', 'lane 2', 'lane 3'],
        ['拥有的两行head', '0 和 8', '0 和 8', '0 和 8', '0 和 8'],
        ['第0个8列组', 'token 0,1', 'token 2,3', 'token 4,5', 'token 6,7'],
        ['第1个8列组', 'token 8,9', 'token 10,11', 'token 12,13', 'token 14,15'],
    ], rh=42)
    d.text(40, 987, 'w 为 WG 内 warp 编号0..3；producer行号还需加32×CTA编号。', 19, MUTED)
    d.text(40, 1020, 'MMA 的64行是 heads，N轴是 selected tokens；每个warp的结果覆盖16个heads。', 19, MUTED)
    return d


def addresses():
    d = Drawing('G2S 的两套地址：INTER 与 SW128',
                'global byte地址与shared BF16元素地址分开计算；以下shared公式均为相对当前缓冲槽的offset。', 1130)
    box(d, 40, 112, 1120, 112, 'Global cache record：656 bytes',
        ['[0,512)：512×E4M3     [512,528)：4×FP32 scale     [528,656)：64×BF16 RoPE',
         'row_base = cache + (slot / page_size) × page_stride + (slot % page_size) × 656'])
    box(d, 40, 254, 480, 121, 'lane5：selected row 5，feature 0..15',
        ['global：一次16-byte load → 16个FP8', 'register：解量化 → 16个BF16，32 bytes'], BLUE_LIGHT)
    arrow(d, 520, 310, 605, 310)
    box(d, 615, 254, 545, 121, 'shared：两次16-byte store',
        ['feature 0..7 → BF16 offset 40..47', 'feature 8..15 → BF16 offset 552..559'], GOLD_LIGHT)
    d.text(40, 421, 'INTER：72个64×8条带；一个8值向量内部连续。', 22, bold=True)
    for j in range(4):
        x = 40+j*285
        box(d, x, 443, 265, 118, f'feature {j*8}..{j*8+7}',
            [f'64 rows × 8 BF16', f'条带基址 {j*512}'], BLUE_LIGHT if j % 2 == 0 else GOLD_LIGHT)
    d.text(40, 598, 'I(r,f) = 512 × floor(f/8) + 8r + (f mod 8)', 23, BLUE, True)
    d.text(40, 640, 'SW128：先分64×64条带，再将每行的8个16-byte包做 XOR 置换。', 22, bold=True)
    d.text(40, 673, '下表是逻辑包 g=0..7 写到的物理包编号；包内8个BF16不变。', 19, MUTED)
    d.table(40, 697, [160]+[120]*8, [
        ['row mod 8']+[f'g={g}' for g in range(8)],
        *[[str(r)]+[str(g ^ r) for g in range(8)] for r in range(8)]
    ], rh=34, highlights=(1, 2))
    d.text(40, 1042, 'W(r,f) = 4096 × floor(f/64) + 64r + ((f mod 64) XOR (8 × (r mod 8)))', 20, GOLD, True)
    d.text(40, 1083, 'SW128用于 Q / P / O，以及BF16 prefill的K；FP8 decode的K使用上面的INTER。', 19, MUTED)
    return d


def handoff():
    d = Drawing('MMA、P的R2S，以及O的S2G',
                'SS：两个矩阵操作数都在shared；RS：A在register、B在shared。箭头表示数据关系，不表示时长。', 1070)
    box(d, 40, 112, 320, 122, 'Q shared · SW128', ['64 heads × 576', 'G2S：TMA'], BLUE_LIGHT)
    box(d, 420, 112, 320, 122, 'K shared · INTER', ['64 tokens × 576', 'G2R → dequant → R2S'], GOLD_LIGHT)
    box(d, 800, 112, 360, 122, 'WG0 · QK：SS WGMMA', ['m64n64k16 × 36', 'rScore：32个FP32/thread'])
    d.text(390, 184, '×', 28, anchor='middle')
    arrow(d, 740, 173, 790, 173)
    d.text(40, 282, '矩阵无需显式ldmatrix到线程寄存器；WGMMA使用shared descriptor读Q/K。', 21, bold=True)
    d.text(40, 327, 'softmax后：rP有32个BF16/thread；同一批值可用于RS，或用STSM写shared。', 21, bold=True)
    d.table(40, 351, [270, 280, 285, 285], [
        ['同一线程，warp0 lane0', '逻辑矩阵坐标', 'BF16本地槽', 'PV的k16包'],
        ['第0个8列组', '(0,0),(0,1),(8,0),(8,1)', '0,1,2,3', '包0的前4值'],
        ['第1个8列组', '(0,8),(0,9),(8,8),(8,9)', '4,5,6,7', '包0的后4值'],
        ['第2/3个8列组', '同两行，列16..31的一部分', '8..15', '包1'],
        ['第4/5；6/7个8列组', '同两行，列32..63的一部分', '16..23；24..31', '包2；包3'],
    ], rh=43)
    box(d, 40, 599, 535, 151, 'WG0：RS PV → O左半',
        ['rP[64,64] × V左[64,256]', 'P寄存器直接作为A，V使用K前256维view', 'm64n256k16 × 4 → 128个FP32/thread'], BLUE_LIGHT)
    box(d, 625, 599, 535, 151, 'WG1：SS PV → O右半',
        ['STSM将P写入SW128 shared → fence / barrier', 'sP[64,64] × V右[64,256]；同时接收rescale', 'm64n256k16 × 4 → 128个FP32/thread'], GOLD_LIGHT)
    d.text(40, 793, 'V(f,t) 与 K(t,f) 共用地址；右半V的基址 +16384 BF16，无须真实转置。', 21, bold=True)
    box(d, 40, 824, 1120, 119, '输出：O register FP32 → 归一化 / BF16 → STSM → shared SW128 → TMA → global',
        ['每WG：64×256；每线程128值，分16包，每包8个BF16，用16轮x4 STSM。',
         'shared：按64×64条带并swizzle；global：每head的512维连续，由TMA descriptor连接。'])
    d.text(40, 986, '有split：R2S保存FP32，shared offset = 520×head + feature；逐行bulk S2G。', 20, GOLD)
    d.text(40, 1027, 'retile只改变同一线程的索引视图；BF16转换、STSM、同步分别执行数值转换、搬运与交接。', 19, MUTED)
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, build in [('flashmla_thread_map', ownership), ('flashmla_shared_address', addresses),
                        ('flashmla_mma_handoff', handoff)]:
        d = build()
        path = args.output / (name+'.svg')
        path.write_text(d.svg(), encoding='utf-8')
        renderer.W, renderer.H = 1200, d.height
        renderer.render_png(path, path.with_suffix('.png'))


if __name__ == '__main__':
    main()
