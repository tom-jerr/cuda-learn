#!/usr/bin/env python3
"""Tile-first CuTe diagrams: G2S source steps, MMA storage, and S2R grouping.

Source of truth: the fixed half_gemm::Config and multi_stage_layout.cu checks.
Run without arguments beside this file, or provide a different output directory.
"""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import cute_r2s_retile as renderer

INK, MUTED, GRID = '#172033', '#5D687A', '#CBD3DE'
PANEL, BLUE, BLUE_LIGHT = '#F7F9FC', '#174EA6', '#DCE8FF'
GOLD, GOLD_LIGHT = '#995600', '#FCE9C8'

class Drawing:
    def __init__(self, title, subtitle, height):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(subtitle)}</desc>',
                      '<style>text{font-family:Arial,"Droid Sans Fallback","Noto Sans CJK SC",sans-serif}.mono{font-family:"DejaVu Sans Mono","Droid Sans Fallback",monospace}</style>']
        self.rect(0, 0, 1200, height, 'white', 'white')
        self.text(40, 45, title, 29, bold=True)
        self.text(40, 79, subtitle, 18, MUTED)
    def rect(self, x, y, w, h, fill='white', stroke=GRID, sw=1):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self, x, y, value, size=19, color=INK, bold=False, anchor='start', mono=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}"'+(' class="mono"' if mono else '')+f'>{html.escape(str(value))}</text>')
    def table(self, x, y, widths, rows, rh=42, highlights=()):
        for r, row in enumerate(rows):
            cx = x
            for c, (value, width) in enumerate(zip(row, widths)):
                fill = PANEL if r == 0 else BLUE_LIGHT if r in highlights else 'white'
                self.rect(cx, y+r*rh, width, rh, fill)
                self.text(cx+12, y+r*rh+rh/2+7, value, 18, bold=(r == 0))
                cx += width
    def svg(self):
        return '\n'.join(self.parts + ['</svg>']) + '\n'

def g2s():
    d = Drawing('G2S：先数 32×32 tile，再比较源与目的地址',
                'A(M,K) 与 B(N,K) 同理。所有地址差以 half 为单位；K=256，shared 每行逻辑宽32。', 960)
    d.text(40, 130, 'global：128×256，被切成 4×8 个 copy tile', 22, bold=True)
    d.text(720, 130, 'shared：3 个 128×32 stage', 22, bold=True)
    for c in range(8):
        d.text(90+c*64+32, 188, f'kt={c}', 15, anchor='middle', mono=True)
    for r in range(4):
        d.text(78, 239+r*52, f'r{r}', 17, anchor='end')
        for c in range(8):
            d.rect(90+c*64, 205+r*52, 64, 52, BLUE_LIGHT if c == 0 else GOLD_LIGHT if c == 1 else 'white')
            d.text(122+c*64, 237+r*52, '32×32', 15, anchor='middle')
    for st in range(3):
        d.text(735+st*120+60, 188, f'stage {st}', 17, anchor='middle')
        for r in range(4):
            d.rect(735+st*120, 205+r*52, 120, 52, BLUE_LIGHT if st == 0 else GOLD_LIGHT if st == 1 else PANEL)
            d.text(795+st*120, 237+r*52, '32×32', 18, anchor='middle')
    d.text(40, 451, '向下32行：跨 32 条完整 global 行 → +8192', 20, BLUE)
    d.text(720, 451, '向下32行 → shared +1024', 20, BLUE)
    d.table(40, 490, [285, 300, 535], [
        ['线程 view 中的移动', 'global 源地址差', 'shared 目的地址差'],
        ['向量内向右1个 half', '+1', '+1（swizzle 保留8-half向量内部）'],
        ['向下一个 copy tile', '+8192：跨32行，每行256', '+1024：跨32行，每行32'],
        ['向右一个 global K tile', '+32：仍在同一组 global 行', '由循环选 stage，不是向右分配新空间'],
        ['换到下一个 shared stage', '由循环选 K tile', '+4096：一个完整128×32缓冲槽'],
    ], rh=48, highlights=(2,))
    d.text(40, 779, 'shape：每线程8个值 × 向下4块 × 块内K方向1块 × 8个K tiles / 3个stages', 21, bold=True)
    d.text(40, 824, 'src: ((8,1),4,1,8):((1,0),8192,0,32)', 23, mono=True)
    d.text(40, 868, 'dst: 向量 +1；向下 +1024；换 stage +4096', 23, mono=True)
    d.text(40, 917, 'shape 中大小为1的mode没有第二个位置；对应stride常打印为0。', 19, MUTED)
    return d

def mma():
    d = Drawing('MMA：tile 的排列决定数量，存储表决定 stride',
                '下面的槽位范围属于同一个线程，单位为 half；不是全局地址，也不是机器寄存器 R 编号。', 1080)
    d.table(40, 112, [120, 290, 235, 255, 220], [
        ['操作数', '4个warp各做一次atom', '每线程基本值数', '需要重复的tile数', 'fragment shape'],
        ['A', '32×16（M×K）', '8 half', 'M方向4；K方向2', '(8,4,2)'],
        ['B', '16×16（N×K）', '4 half', 'N方向8；K方向2', '(4,8,2)'],
        ['C', '32×16（M×N）', '4 half', 'M方向4；N方向8', '(4,4,8)'],
    ], rh=40)
    d.text(40, 322, 'A：一个M位置的两段k16相邻存放', 22, bold=True)
    d.text(640, 322, 'B：一个N位置的两段k16相邻存放', 22, bold=True)
    d.table(40, 347, [180, 165, 165], [['A 的 M tile', 'K0..15', 'K16..31']] +
            [[f'M组{i}', f'{16*i}..{16*i+7}', f'{16*i+8}..{16*i+15}'] for i in range(4)], rh=42)
    d.table(640, 347, [180, 165, 165], [['B 的 N tile', 'K0..15', 'K16..31']] +
            [[f'N组{j}', f'{8*j}..{8*j+3}', f'{8*j+4}..{8*j+7}'] for j in range(4)], rh=42)
    d.text(40, 591, '下移M32：+16槽；右移K16：+8槽', 20, BLUE)
    d.text(640, 591, '下移N16：+8槽；右移K16：+4槽', 20, BLUE)
    d.text(640, 618, 'B 共8个N组，此处只画前4个。', 17, MUTED)
    d.text(40, 670, 'C：每个N分组，先存完沿M向下的4个小tile', 22, bold=True)
    rows = [['C 的行范围', 'N0..15', 'N16..31', 'N32..47', 'N48..63']]
    for i in range(4):
        rows.append([f'M{32*i}..{32*i+31}'] + [f'{4*i+16*j}..{4*i+16*j+3}' for j in range(4)])
    d.table(40, 697, [220, 225, 225, 225, 225], rows, rh=43, highlights=(1,))
    d.text(40, 953, 'C 向下32行：槽位 +4；向右16列：槽位 +16。', 22, BLUE, bold=True)
    d.text(40, 996, '把相邻两个N小tile合成32×32宏块后：跨M仍 +4，跨N变为 +32。', 21)
    d.text(40, 1037, '这些是本例 partition_fragment 的实测存储顺序；shape 本身不能唯一确定 stride。', 18, MUTED)
    return d

def s2r():
    d = Drawing('S2R：x4 按四块8×8加载，retile 按原槽位接收',
                '源侧每个lane提供一行地址；目的侧每个lane接收8个half。源地址提供者与接收者的分工不同。', 1060)
    d.text(40, 131, 'A：一个16×16子块', 23, bold=True)
    d.text(640, 131, 'B：同一warp的两个8×16子块', 23, bold=True)
    blocks_a=[(0,0,'word0','M0..7, K0..7','lanes 0..7'),(0,1,'word1','M8..15, K0..7','lanes 8..15'),
              (1,0,'word2','M0..7, K8..15','lanes 16..23'),(1,1,'word3','M8..15, K8..15','lanes 24..31')]
    blocks_b=[(0,0,'word0','N0..7, K0..7','lanes 0..7'),(1,0,'word1','N0..7, K8..15','lanes 8..15'),
              (0,1,'word2','N16..23, K0..7','lanes 16..23'),(1,1,'word3','N16..23, K8..15','lanes 24..31')]
    for ox, blocks, fill, color in [(40,blocks_a,BLUE_LIGHT,BLUE),(640,blocks_b,GOLD_LIGHT,GOLD)]:
        for c,r,word,coord,lanes in blocks:
            x=ox+c*260;y=158+r*115
            d.rect(x,y,250,105,fill)
            d.text(x+16,y+28,'矩阵'+word[-1]+' → '+word,21,color,True)
            d.text(x+16,y+58,coord,18,mono=True)
            d.text(x+16,y+87,'行首来自 '+lanes,17)
    d.text(40, 410, '每lane：4个word =8half，匹配一个A atom fragment。', 18)
    d.text(640, 410, '每lane：前4half属于一个N子块，后4half属于另一个。', 18)
    d.text(40, 466, 'A：原(8,4,2) → copy(8,4,2)', 23, BLUE, True)
    d.text(640, 466, 'B：原(4,8,2) → copy(8,4,2)', 23, GOLD, True)
    d.table(40, 491, [170,170,170], [['A 的k16', '前4个值的槽位', '后4个值的槽位'],['K0..15','0..3','4..7'],['K16..31','8..11','12..15']],rh=44,highlights=(1,))
    d.table(640, 491, [170,170,170], [['B 的k16', '第一个N子块', '第二个N子块'],['K0..15','0..3','8..11'],['K16..31','4..7','12..15']],rh=44,highlights=(1,))
    d.text(40, 661, '上表固定最前一组。A的8个值连续；B的两组4个值相隔8槽，retile不重新打包。', 21)
    d.table(40, 704, [310,300,255,255], [
        ['copy view 的移动', 'shared 源行首地址差', 'A 寄存器目的槽位差', 'B 寄存器目的槽位差'],
        ['下移32行：A沿M / B沿N', '+1024 half', '+16 half', '+16 half'],
        ['沿K跨16列', '+16 或 -16（swizzle）', '+8 half', '+4 half'],
        ['下一个 shared stage', '+4096 half', '没有stage维', '没有stage维'],
    ],rh=48)
    d.text(40, 944, '源shape=(8,4,2,3)：8个源描述值、4组、2段k16、3个stage。', 21)
    d.text(40, 983, '目的shape=(8,4,2)：每次x4接收8个half；4组 × 2段k16 = 每warp 8次x4。', 21)
    d.text(40, 1022, '示意块的坐标以warp0、首组、首段k16为基准；其他warp/分组再加各自起点。', 18, MUTED)
    return d

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', nargs='?', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in [('cute_g2s_tile_steps',g2s),('cute_mma_tile_storage',mma),('cute_s2r_tile_retile',s2r)]:
        drawing = factory()
        path = args.output_dir / (name+'.svg')
        path.write_text(drawing.svg(),encoding='utf-8')
        previous_size = renderer.W, renderer.H
        try:
            renderer.W, renderer.H = 1200, drawing.height
            renderer.render_png(path, path.with_suffix('.png'))
        finally:
            renderer.W, renderer.H = previous_size

if __name__ == '__main__':
    main()
