#!/usr/bin/env python3
"""Figures for the pasted BM=BN=64, BK=32 BF16 GEMM. Optional output directory.

Logical coordinates come from src/gemm_mma.cu and the supplied source;
fragment layouts follow NVIDIA PTX m16n8k16 BF16 and ldmatrix.m8n8.
Uses the repository's librsvg renderer; no external image assets.
"""
import argparse
import html
from pathlib import Path
import cute_r2s_retile as renderer

INK, MUTED, GRID = '#172033', '#5D687A', '#CBD3DE'
BLUE, BL = '#174EA6', '#DCE8FF'
GOLD, GL = '#995600', '#FCE9C8'
GREEN, GR = '#21654D', '#DEF1E8'
PURPLE, PL = '#7142A0', '#EEE4F7'
COLORS = [BL, GL, GR, PL]

class Figure:
    def __init__(self, title, subtitle, height):
        self.height = height
        self.p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
                  f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(subtitle)}</desc>',
                  '<style>text{font-family:Arial,"Droid Sans Fallback","Noto Sans CJK SC",sans-serif}</style>']
        self.rect(0, 0, 1200, height, 'white', 'white')
        self.text(40, 46, title, 29, bold=True)
        self.text(40, 80, subtitle, 18, MUTED)
    def rect(self, x, y, w, h, fill='white', stroke=GRID, sw=1):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self, x, y, value, size=18, color=INK, bold=False, anchor='start'):
        self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{html.escape(str(value))}</text>')
    def line(self, x1, y1, x2, y2, color=GRID, sw=1, dash=False):
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"'+(' stroke-dasharray="5 5"' if dash else '')+'/>')
    def table(self, x, y, widths, rows, rh=40, size=17):
        for r, row in enumerate(rows):
            cx = x
            for width, value in zip(widths, row):
                self.rect(cx, y+r*rh, width, rh, '#F7F9FC' if r == 0 else 'white')
                self.text(cx+12, y+r*rh+rh/2+6, value, size, bold=r == 0)
                cx += width
    def svg(self):
        return '\n'.join(self.p+['</svg>'])+'\n'

def copy_figure():
    d = Figure('Block copy：搬运分工与 C 的计算分工', '固定一个 bk；以下均为 block 内逻辑坐标。q=0,1 是每个 copy for 循环的两轮。', 1010)
    d.text(40, 128, 'A：64×32，每条色带 = 8 行 × 32 列', 21, bold=True)
    d.text(440, 128, 'B：32×64，每条色带 = 4 行 × 64 列', 21, bold=True)
    for x, operand, step, w in [(40, 'A', 8, 350), (440, 'B', 4, 400)]:
        for q in range(2):
            for warp in range(4):
                i=q*4+warp
                d.rect(x, 152+i*37, w, 37, COLORS[warp])
                d.text(x+12, 176+i*37, f'q={q}  warp {warp}   rows {i*step}..{(i+1)*step-1}', 18)
    d.text(886, 128, 'C：64×64', 21, bold=True)
    for warp in range(4):
        x=886+(warp%2)*137; y=152+(warp//2)*148
        d.rect(x,y,137,148,COLORS[warp])
        d.text(x+68,y+48,f'warp {warp}',21,bold=True,anchor='middle')
        d.text(x+68,y+81,'32×32',21,anchor='middle')
        d.text(x+68,y+112,f'({warp//2},{warp%2})',17,anchor='middle')
    d.text(40, 491, '每线程每轮搬 8 个 BF16 = 16 B；一轮全 block 搬 128×8 = 1024 个值。', 21, bold=True)
    d.table(40,515,[145,395,325,255],[
        ['对象','linear = 8×tid + 1024×q','row','col（连续搬8个值）'],
        ['A','tid = 32×warp + lane','8×warp + lane/4 + 32×q','8×(lane % 4)'],
        ['B','整数除法均向下取整','4×warp + lane/8 + 16×q','8×(lane % 8)'],
    ],rh=44)
    d.text(40, 691, 'warp 0，q=0：lane 如何铺满第一行？', 22, bold=True)
    d.text(40,733,'A row 0',18,bold=True)
    for lane in range(4):
        x=170+lane*120
        d.rect(x,705,120,48,BL)
        d.text(x+60,726,f'lane {lane}',16,anchor='middle')
        d.text(x+60,746,f'K {8*lane}..{8*lane+7}',15,anchor='middle')
    d.text(685,737,'lane 4..7 → row 1；…；lane 28..31 → row 7',17)
    d.text(40,799,'B row 0',18,bold=True)
    for lane in range(8):
        x=170+lane*120
        d.rect(x,771,120,48,GL)
        d.text(x+60,792,f'lane {lane}',16,anchor='middle')
        d.text(x+60,812,f'N {8*lane}..{8*lane+7}',15,anchor='middle')
    d.text(40, 857, 'q=1：A 整体下移32行，B 整体下移16行；warp、lane 不变。', 20)
    d.text(40, 898, '物理 shared stride：A=40 个 BF16（80 B），B=72 个 BF16（144 B）。PAD 不参与逻辑 tile。',18)
    d.text(40, 939, 'copy 后全 block 同步；warp 0/1 复用同一片 A，warp 0/2 复用同一片 B。',20,BLUE,bold=True)
    d.text(40, 977, 'cp.async 的数据直接进入 shared；copy 的源数据不经过程序可见的 fragment 寄存器。',18,MUTED)
    return d

def a_figure():
    d=Figure('A：ldmatrix.x4 的行地址与寄存器归属', '固定 warp、mi、kk。M0=32×warp_m+16×mi；g=lane/4，t=lane%4。', 1000)
    d.text(40,126,'① 四个 8×8 子矩阵：提供行地址的 lane',21,bold=True)
    for j,(r,c) in enumerate([(0,0),(8,0),(0,8),(8,8)]):
        x=40+(c//8)*255; y=150+(r//8)*102
        d.rect(x,y,255,102,COLORS[j])
        d.text(x+12,y+29,f'Q{j} → a_frag[mi][{j}]',19,bold=True)
        d.text(x+12,y+57,f'行 {r}..{r+7}，K {c}..{c+7}',17)
        d.text(x+12,y+83,f'地址来自 lane {j*8}..{j*8+7}',17)
    d.text(610,126,'② 每个 lane 都收到四个 32-bit 寄存器',21,bold=True)
    d.table(610,150,[110,440],[
        ['寄存器','低16位 / 高16位：A_smem 坐标'],
        ['a[0]','(M0+g, kk+2t) / (M0+g, kk+2t+1)'],
        ['a[1]','(M0+g+8, kk+2t) / (M0+g+8, kk+2t+1)'],
        ['a[2]','(M0+g, kk+2t+8) / (M0+g, kk+2t+9)'],
        ['a[3]','(M0+g+8, kk+2t+8) / (M0+g+8, kk+2t+9)'],
    ],rh=41,size=16)
    d.text(40,408,'③ 完整 16×16 A fragment：每格是横向相邻的两个 BF16',21,bold=True)
    d.text(40,437,'标签 Lx / aj：lane x 的 a_frag[mi][j]；行、列均为 fragment 内相对坐标。',17,MUTED)
    x,y,cw,ch=80,476,62,27
    for c in range(8): d.text(x+(c+.5)*cw,y-12,f'{2*c},{2*c+1}',14,anchor='middle')
    for r in range(16):
        d.text(x-12,y+r*ch+19,r,15,anchor='end')
        for c in range(8):
            j=(r//8)+2*(c//4); lane=4*(r%8)+c%4
            d.rect(x+c*cw,y+r*ch,cw,ch,COLORS[j],GRID)
            d.text(x+(c+.5)*cw,y+r*ch+18,f'L{lane} / a{j}',13,anchor='middle')
    d.text(635,490,'以 lane 13 为例：g=3，t=1',21,bold=True)
    for i,line in enumerate([
        '它提供的行地址：A_smem[M0+13][kk]。',
        '这个地址属于 Q1 的行索引5（从0开始）。',
        '但它最终收到的是：',
        'a[0] = A[M0+3,  kk+2 : kk+3]',
        'a[1] = A[M0+11, kk+2 : kk+3]',
        'a[2] = A[M0+3,  kk+10 : kk+11]',
        'a[3] = A[M0+11, kk+10 : kk+11]',
        '冒号表示含两端的相邻元素。',
    ]): d.text(635,531+37*i,line,18)
    d.text(635,858,'32 lanes × 4 regs × 2 BF16 = 16×16',19,BLUE,bold=True)
    d.text(40,962,'mi=0,1：沿 M 加16；kk=0,16：沿 K 加16。x4 是一条 warp 指令中的四个 8×8 子矩阵。',20,BLUE,bold=True)
    return d

def b_figure():
    d=Figure('B：ldmatrix.x2.trans 把竖向两个值装进一个 reg', '固定 warp、ni、kk。N0=32×warp_n+8×ni；g=lane/4，t=lane%4。', 950)
    d.text(40,126,'① shared 中仍然是 row-major B[16][8]',21,bold=True)
    for j in range(2):
        d.rect(40,150+j*97,485,97,COLORS[j])
        d.text(55,178+j*97,f'Q{j}：K {8*j}..{8*j+7}，N 0..7',20,bold=True)
        d.text(55,211+j*97,f'行地址来自 lane {8*j}..{8*j+7} → b_frag[ni][{j}]',18)
    d.text(590,126,'② .trans 改变装入寄存器的排列',21,bold=True)
    d.table(590,150,[100,470],[
        ['寄存器','低16位 / 高16位：B_smem 坐标'],
        ['b[0]','(kk+2t, N0+g) / (kk+2t+1, N0+g)'],
        ['b[1]','(kk+2t+8, N0+g) / (kk+2t+9, N0+g)'],
    ],rh=47,size=17)
    d.text(590,322,'原矩阵同一列、相邻两行 → 一个 32-bit reg。',19,BLUE,bold=True)
    d.text(40,394,'③ 完整 16×8 B fragment：每格包含竖向两个 BF16',21,bold=True)
    x,y,cw,ch=92,443,54,49
    for c in range(8): d.text(x+(c+.5)*cw,y-12,c,16,anchor='middle')
    for rp in range(8):
        d.text(x-12,y+rp*ch+30,f'{rp*2},{rp*2+1}',15,anchor='end')
        for c in range(8):
            lane=c*4+rp%4; j=rp//4
            d.rect(x+c*cw,y+rp*ch,cw,ch,COLORS[j])
            d.text(x+(c+.5)*cw,y+rp*ch+21,f'L{lane}',16,anchor='middle')
            d.text(x+(c+.5)*cw,y+rp*ch+41,f'b{j}',14,anchor='middle')
    d.text(590,458,'lane 13：g=3，t=1',21,bold=True)
    lines=[
        '提供地址：B_smem[kk+13][N0]。',
        '收到 b[0]：B[kk+2,N0+3]，B[kk+3,N0+3]',
        '收到 b[1]：B[kk+10,N0+3]，B[kk+11,N0+3]',
        '',
        'x2 仅使用 lane 0..15 提供的行地址。',
        'lane 16..31 仍参与指令，并收到两个 reg。',
        '代码中 +(lane>>4) 对有效的16个地址恒为0，',
        '它不负责转置；转置来自 .trans。',
    ]
    for i,line in enumerate(lines): d.text(590,499+i*36,line,18)
    d.text(590,825,'32 lanes × 2 regs × 2 BF16 = 16×8',20,BLUE,bold=True)
    d.text(40,892,'ni=0,1,2,3：沿 N 每次加8；kk=0,16：沿 K 加16。shared 中没有原地转置操作。',20,BLUE,bold=True)
    d.text(40,929,'.x2 的地址来源规则按本例 Ampere / sm_80 解释；所有32个lane必须执行同一条 ldmatrix。',17,MUTED)
    return d

def mma_figure():
    d=Figure('MMA：一个 warp 的 8 个 tile 与每 lane 的 32 个累加器', 'MMA atom：A[16×16] × B[16×8] → C[16×8]；每个 kk 对同一份 accum 追加累加。', 1020)
    d.text(40,126,'① warp 的 C[32×32]：mi 外层、ni 内层',21,bold=True)
    for mi in range(2):
        for ni in range(4):
            x=40+ni*134; y=153+mi*95
            d.rect(x,y,134,95,COLORS[ni])
            d.text(x+67,y+28,f'#{mi*4+ni}  16×8',18,bold=True,anchor='middle')
            d.text(x+67,y+57,f'mi={mi}, ni={ni}',17,anchor='middle')
            d.text(x+67,y+82,f'accum[{mi}][{ni}]',15,anchor='middle')
    d.text(625,126,'② fragment 复用与线程私有数组',21,bold=True)
    for i,line in enumerate([
        'a_frag[mi] 被 ni=0..3 的4次 MMA 复用。',
        'b_frag[ni] 被 mi=0..1 的2次 MMA 复用。',
        '每 lane：a_frag[2][4] = 8 个32-bit槽位',
        '每 lane：b_frag[4][2] = 8 个32-bit槽位',
        '每 lane：accum[2][4][4] = 32 个FP32槽位',
    ]): d.text(625,168+i*39,line,19)
    d.text(40,397,'③ 一个 16×8 C tile 的完整分布（4个连续lane共同覆盖一行）',21,bold=True)
    x,y,cw,ch=80,444,111,26
    for c in range(4): d.text(x+(c+.5)*cw,y-12,f'N {2*c},{2*c+1}',16,anchor='middle')
    for r in range(16):
        d.text(x-12,y+r*ch+18,r,15,anchor='end')
        for c in range(4):
            lane=4*(r%8)+c; j=2*(r//8)
            d.rect(x+c*cw,y+r*ch,cw,ch,BL if r<8 else GL)
            d.text(x+(c+.5)*cw,y+r*ch+18,f'L{lane}: d{j},d{j+1}',14,anchor='middle')
    d.text(625,456,'令 M0=32×warp_m+16×mi',20,bold=True)
    d.text(625,489,'令 N0=32×warp_n+8×ni',20,bold=True)
    d.table(625,513,[115,420],[
        ['FP32 reg','对应 C 的 block 内坐标'],
        ['d[0]','(M0+g, N0+2t)'],
        ['d[1]','(M0+g, N0+2t+1)'],
        ['d[2]','(M0+g+8, N0+2t)'],
        ['d[3]','(M0+g+8, N0+2t+1)'],
    ],rh=42)
    d.text(625,765,'g=lane/4；t=lane%4',20,BLUE,bold=True)
    d.text(625,804,'加 block_m / block_n 后就是 HBM 坐标。',18)
    d.text(625,843,'d[0..3] 即 accum[mi][ni][0..3]。',18)
    d.text(40,909,'32 lanes × 4 FP32 = 128 个值 = 一个16×8 tile；再 ×8 个tile = warp 的32×32 C。',20,BLUE,bold=True)
    d.text(40,951,'寄存器属于 lane：同名 accum[0][0][0] 在32个lane中有32份独立的值。',20,bold=True)
    d.text(40,990,'图中是逻辑槽位；物理 R 编号、实际寄存器用量与 spill 需查看编译器输出。MMA 由整个 warp 协作。',18,MUTED)
    return d

def schedule_figure():
    d=Figure('迭代顺序：每个 bk 搬32层 K，再分两次 k16 计算', '示意顺序，宽度不代表耗时；warp 间的计算进度可不同。当前代码没有跨 bk 的 copy / MMA 流水重叠。', 750)
    steps=[(180,170,'copy issue'),(350,88,'commit'),(438,106,'wait_all'),(570,190,'kk=0'),(785,190,'kk=16')]
    for warp in range(4):
        y=142+warp*63
        d.text(40,y+31,f'warp {warp}',20,bold=True)
        for x,w,label in steps:
            d.rect(x,y,w-8,47,COLORS[warp] if label.startswith('kk') else '#F7F9FC')
            d.text(x+(w-8)/2,y+29,label,17,anchor='middle')
    for x,label in [(551,'CTA barrier'),(996,'CTA barrier')]:
        d.line(x,122,x,393,BLUE,2,True)
        d.text(x,112,label,16,BLUE,anchor='middle')
    d.text(1018,198,'下一轮',18,bold=True)
    d.text(1018,232,'bk += 32',18)
    d.text(1018,266,'覆盖 shared',17)
    d.line(180,407,544,407,MUTED,2)
    d.text(180,434,'异步 copy：issue 后可在途；wait_all 返回时该线程的 copy 已完成。',18,MUTED)
    d.text(40,486,'展开任意一个 kk（每 warp）：',22,bold=True)
    labels=[('A load', 'mi=0,1', '2× ldmatrix.x4'),('B load','ni=0,1,2,3','4× ldmatrix.x2.trans'),('MMA','(mi,ni)=(0,0)…(1,3)','8× mma.m16n8k16')]
    for i,(title,sub,detail) in enumerate(labels):
        x=40+i*383
        d.rect(x,510,354,105,COLORS[i])
        d.text(x+16,540,title,21,bold=True)
        d.text(x+16,569,sub,18)
        d.text(x+16,597,detail,18)
        if i<2: d.text(x+361,568,'→',21)
    d.text(40,658,'每 bk / 每 warp：4次 x4 + 8次 x2.trans + 16次 MMA；全 block：64次 warp 级 MMA。',20,bold=True)
    d.text(40,700,'accum：初始化一次 → 累加 bk+kk 的每段 K → 全部 bk 完成后转 BF16、写回 HBM。',20,BLUE,bold=True)
    d.text(40,735,'两处 __syncthreads()：前者保证可读取全 block 的 copy 结果；后者保证 shared 被覆盖前所有 warp 已用完。',17,MUTED)
    return d

def verify_mapping():
    # Exhaustive logical coverage: every output/copy element has exactly one owner.
    for rows,cols in [(64,32),(32,64)]:
        coords=[]
        for tid in range(128):
            for q in range(2):
                linear=tid*8+q*1024
                coords.extend((linear//cols,linear%cols+i) for i in range(8))
        assert len(coords)==len(set(coords))==rows*cols
        assert set(coords)=={(r,c) for r in range(rows) for c in range(cols)}
    out=[]
    for warp in range(4):
        for lane in range(32):
            g,t=divmod(lane,4)
            for mi in range(2):
                for ni in range(4):
                    out.extend((32*(warp//2)+16*mi+g+8*(v//2),32*(warp%2)+8*ni+2*t+v%2) for v in range(4))
    assert len(out)==len(set(out))==4096
    for operand in ['A','B']:
        coords=[]
        for lane in range(32):
            g,t=divmod(lane,4)
            if operand=='A':
                coords.extend((g+8*(j%2),2*t+8*(j//2)+h) for j in range(4) for h in range(2))
            else:
                coords.extend((2*t+8*j+h,g) for j in range(2) for h in range(2))
        assert len(coords)==len(set(coords))==(256 if operand=='A' else 128)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=Path(__file__).parent)
    parser.add_argument('--svg-only',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    verify_mapping()
    for name,fn in [('copy',copy_figure),('a',a_figure),('b',b_figure),('mma',mma_figure),('schedule',schedule_figure)]:
        d=fn(); path=args.output/f'ampere_{name}.svg'; path.write_text(d.svg())
        if not args.svg_only:
            renderer.W,renderer.H=1200,d.height
            renderer.render_png(path,path.with_suffix('.png'))
    print('Generated 5 figures; exhaustive copy and fragment ownership checks passed.')

if __name__=='__main__': main()
