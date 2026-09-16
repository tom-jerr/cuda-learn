#!/usr/bin/env python3
"""Editable tile diagrams for the pinned FA3 / FlashMLA source walkthrough.

Geometry is schematic, never a performance measurement. Optional positional
argument selects an output directory. Numbers are checked by hopper_layout_probe.cu.
"""
import argparse
from pathlib import Path
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD, GOLD_LIGHT, PANEL, MUTED
import cute_r2s_retile as renderer

def box(d,x,y,w,h,title,lines,fill=PANEL):
    d.rect(x,y,w,h,fill)
    d.text(x+16,y+30,title,21,bold=True)
    for i,line in enumerate(lines): d.text(x+16,y+60+29*i,line,18)

def arrow(d,x,y,x2,y2):
    d.parts.append(f'<path d="M{x},{y} L{x2},{y2}" fill="none" stroke="#172033" stroke-width="2"/>')
    if y==y2:
        sign=1 if x2>x else -1
        pts=f'{x2},{y2} {x2-9*sign},{y2-5} {x2-9*sign},{y2+5}'
    else:
        sign=1 if y2>y else -1
        pts=f'{x2},{y2} {x2-5},{y2-9*sign} {x2+5},{y2-9*sign}'
    d.parts.append(f'<polygon points="{pts}" fill="#172033"/>')

def fa3():
    d=Drawing('FA3 FP8：先确定两个 GEMM 的 tile',
              '固定 D=Dv=128、V 行优先：CTA 的 M=128，N=224；两个 consumer WG，每组128线程。',1000)
    box(d,40,112,325,158,'Q：128 query × 128 D',
        ['WG0：query 0..63','WG1：query 64..127','FP8；TMA → shared'],BLUE_LIGHT)
    box(d,420,112,325,158,'K：224 token × 128 D',
        ['两个 WG 复用同一个 K tile','D 方向分成 4 段 k32','FP8；TMA → shared'],GOLD_LIGHT)
    box(d,800,112,360,158,'score：128 query × 224 token',
        ['每 WG：64×224','每线程：112 个 FP32','QK：每 WG 发出 4 次 WGMMA'])
    d.text(391,200,'×',32,anchor='middle');arrow(d,745,191,790,191)
    d.text(40,322,'V 的逻辑轴始终是 (Dv, token)；搬运前后改变的是物理排列',23,bold=True)
    box(d,40,348,520,162,'TMA 落地的 Vt：沿 Dv 连续',
        ['每个 token 的 128 FP8：一条128-byte行','(Dv,token,stage) = (128,224,2)','MN-major；SW128'],GOLD_LIGHT)
    box(d,640,348,520,162,'WGMMA 使用的 V：沿 token 分块连续',
        ['沿 token 切成 7 个32列小块','(Dv,token,stage) = (128,224,2)','K-major；SW32；包含配套置换'],BLUE_LIGHT)
    arrow(d,560,429,630,429)
    d.text(40,550,'真实转置：LDSM.T → 寄存器 byte_perm → STSM；单改 stride 不会移动字节。',21,bold=True)
    d.text(40,597,'把 V 分成 64 Dv × 32 token 的小块：2 × 7 = 14 块',22,bold=True)
    for c in range(7):
        d.text(210+c*135+62,637,f'token {c*32}',16,anchor='middle')
        for r in range(2):
            d.rect(210+c*135,650+r*59,125,51,BLUE_LIGHT if r==0 else GOLD_LIGHT)
            d.text(272+c*135,681+r*59,'64×32',20,anchor='middle')
    d.text(40,681,'Dv 0..63',18);d.text(40,740,'Dv 64..127',18)
    d.text(40,817,'每小块 2048 bytes ÷ 128线程 = 16 bytes/thread；两块一批，共7批。',21)
    d.text(40,859,'copy view：16个值 × 2个Dv位置 × 7个token位置 × 2个stage',21)
    d.text(40,907,'PV：每 WG 的 P(64×224) × V(224×128) → O(64×128)',22,BLUE,True)
    d.text(40,952,'P 在寄存器、V 在 shared；224 / k32 = 7 次 WGMMA，64个 FP32/thread 输出。',20)
    return d

def registers():
    d=Drawing('WGMMA accumulator：沿输出列扩展，每跨8列增加4槽',
              '图中“槽”是当前线程的逻辑元素编号，不是机器寄存器编号；128线程协作一个m64 tile。',960)
    d.text(40,132,'固定一个线程：它负责两行，每行每个8列分组中有2个相邻值',23,bold=True)
    d.table(40,160,[235,220,220,220,225],[
        ['该线程持有的行','第0个8列分组','第1个8列分组','第2个8列分组','第3个8列分组'],
        ['本线程的上行','槽 0,1','槽 4,5','槽 8,9','槽 12,13'],
        ['相隔8行的下行','槽 2,3','槽 6,7','槽 10,11','槽 14,15']],rh=51)
    d.text(40,357,'shape ((2,2,N/8),1,1)；stride ((1,2,4),0,0)',25,BLUE,True)
    d.table(40,391,[280,250,270,320],[
        ['结果 tile（单 WG）','列方向分组','每线程元素数','局部形状'],
        ['FA3 score 64×224','28 个8列分组','2×2×28 = 112','((2,2,28),1,1)'],
        ['FA3 output 64×128','16 个8列分组','2×2×16 = 64','((2,2,16),1,1)'],
        ['MLA score 64×64','8 个8列分组','2×2×8 = 32','((2,2,8),1,1)'],
        ['MLA output 64×256','32 个8列分组','2×2×32 = 128','((2,2,32),1,1)']],rh=48)
    d.text(40,693,'P 从“前一个 GEMM 的输出”变成“后一个 GEMM 的 A 输入”',23,bold=True)
    box(d,40,720,535,153,'FA3：FP8 的 k32 包',
        ['每线程16个 FP8 × 7段 = 112个','((4,2,2),1,7)：k32 步长16槽','配套 permute 是实际交换；view 只改索引'],BLUE_LIGHT)
    box(d,625,720,535,153,'FlashMLA：BF16 的 k16 包',
        ['每线程8个 BF16 × 4段 = 32个','((2,2,2),1,4)：k16 步长8槽','本地 P 用 RS；共享给另一 WG 用 STSM'],GOLD_LIGHT)
    d.text(40,923,'原 score 的一段8列 → 4槽；凑成k16 → 8槽；凑成k32 → 16槽。',21,bold=True)
    return d

def decode():
    d=Drawing('FlashMLA decode：从稀疏 FP8 行到 BF16 shared tile',
              'V3.2：128 heads，cluster内两个CTA；每CTA处理64 heads，负责解量化32个选中token。',1100)
    d.text(40,127,'HBM：一个 token 占656 bytes；四个128维量化组共享各自的scale',23,bold=True)
    segments=[(40,760,'512 bytes：FP8 latent',BLUE_LIGHT),(800,155,'16 bytes',GOLD_LIGHT),(955,205,'128 bytes',PANEL)]
    for x,w,label,fill in segments:
        d.rect(x,148,w,66,fill);d.text(x+w/2,188,label,20,anchor='middle')
    d.text(40,246,'offset 0',18);d.text(800,246,'512：4×FP32',18);d.text(955,246,'528：64×BF16',18)
    d.text(40,289,'indices → page / page内行 → 656-byte记录 → 16-byte load → dequant',22,bold=True)
    d.table(40,314,[240,255,300,325],[
        ['同一warp的4个lane','处理同一个token','本次FP8特征范围','解量化后的写入'],
        ['lane 0','选中序号 0','0..15','两条8-BF16向量'],
        ['lane 8','选中序号 0','16..31','两条8-BF16向量'],
        ['lane 16','选中序号 0','32..47','两条8-BF16向量'],
        ['lane 24','选中序号 0','48..63','两条8-BF16向量']],rh=44)
    d.text(40,565,'shared K：64 token × 576 BF16，按 64×8 小块横向排列',23,bold=True)
    for c in range(4):
        x=190+c*242
        d.rect(x,595,222,119,BLUE_LIGHT if c%2==0 else GOLD_LIGHT)
        d.text(x+111,627,f'feature {c*8}..{c*8+7}',19,anchor='middle')
        d.text(x+111,663,'64 token × 8 BF16',19,anchor='middle')
        d.text(x+111,696,f'base +{c*512} 元素',18,anchor='middle')
    d.text(40,629,'token',18);d.text(40,661,'向下+1',18);d.text(40,693,'地址+8',18)
    d.text(40,758,'跨8个feature：+512 BF16；同一token的16个值拆开写到相隔512元素处。',21,bold=True)
    box(d,40,789,525,156,'CTA0：heads 0..63',
        ['本地解量化 token槽 0..31','接收 CTA1 的 token槽 32..63','最终各自拥有完整64×576 shared K'],BLUE_LIGHT)
    box(d,635,789,525,156,'CTA1：heads 64..127',
        ['本地解量化 token槽 32..63','接收 CTA0 的 token槽 0..31','st.async + cluster transaction barrier'],GOLD_LIGHT)
    arrow(d,565,842,625,842);arrow(d,635,906,575,906)
    d.text(40,994,'本地写入 + DSM远端写入：分担解量化；消费者等本地与远端都ready。',21)
    d.text(40,1043,'V 直接复用 K 的前512维：K(token,feature) ↔ V(feature,token)，不再搬一份。',21,BLUE,True)
    return d

def prefill():
    d=Drawing('FlashMLA：prefill 与 decode 的 WG 分工不同',
              '同一个 query token 的64 heads为行；左右指输出latent的0..255 / 256..511。此图无时间比例。',1010)
    d.text(40,128,'Hopper sparse prefill：输入 Q / KV 都是 BF16',23,bold=True)
    box(d,40,153,345,210,'Producer WG2：稀疏 gather',
        ['8线程合作搬一个token的64维','每线程一条16-byte cp.async','每组负责4行；9个64维条带','K0：第一个64-token块','K1：第二个64-token块'])
    box(d,425,153,345,210,'Consumer WG0',
        ['计算 QK0 → softmax P0','本地 P0 × V0左：RS','读取 P1 × V1左：SS','持有 O[:,0..255]','把 P0 scatter 到 shared'],BLUE_LIGHT)
    box(d,810,153,350,210,'Consumer WG1',
        ['计算 QK1 → softmax P1','本地 P1 × V1右：RS','读取 P0 × V0右：SS','持有 O[:,256..511]','把 P1 scatter 到 shared'],GOLD_LIGHT)
    arrow(d,385,245,415,245);arrow(d,770,245,800,245)
    d.text(40,408,'两段token的 P 要交换，两个输出半块都必须累加两段token；还要协调行max与sum。',21)
    d.table(40,440,[335,390,395],[
        ['prefill copy view 的移动','global 源地址变化（BF16）','shared 目的地址变化（BF16）'],
        ['沿feature跨64维','+64','+4096：跨一个64×64条带'],
        ['同组下一个token槽（+16行）','重新读取indices；不是固定stride','+1024：16行×64'],
        ['换到下一个KV缓冲槽','选下一批indices','+36864：64×576']],rh=49)
    d.text(40,693,'Hopper sparse decode：Q为BF16，KV为656-byte FP8混合记录',23,bold=True)
    box(d,40,718,345,181,'Producer WG2',
        ['indices → gather → dequant','BF16写入本地shared与DSM','每次准备一个64-token块','两个KV缓冲槽循环使用'])
    box(d,425,718,345,181,'Consumer WG0',
        ['计算全部 QK → softmax P','本地 P × V左：RS','把 P 和 rescale 交给 WG1','持有 O[:,0..255]'],BLUE_LIGHT)
    box(d,810,718,350,181,'Consumer WG1',
        ['接收同一份 P 与 rescale','shared P × V右：SS','不重复计算 QK / softmax','持有 O[:,256..511]'],GOLD_LIGHT)
    arrow(d,385,813,415,813);arrow(d,770,813,800,813)
    d.text(40,958,'输出：FP32 accumulator → BF16 + STSM → shared → TMA store；split时另走FP32 combine。',20)
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir',nargs='?',type=Path,default=Path(__file__).parent)
    out=p.parse_args().output_dir;out.mkdir(parents=True,exist_ok=True)
    for name,fn in [('fa3_fp8_tiles',fa3),('hopper_wgmma_registers',registers),('flashmla_fp8_gather',decode),('flashmla_prefill_decode',prefill)]:
        d=fn();path=out/(name+'.svg');path.write_text(d.svg(),encoding='utf-8')
        previous=renderer.W,renderer.H
        try:
            renderer.W,renderer.H=1200,d.height
            renderer.render_png(path,path.with_suffix('.png'))
        finally: renderer.W,renderer.H=previous

if __name__=='__main__': main()
