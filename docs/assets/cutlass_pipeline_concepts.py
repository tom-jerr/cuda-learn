#!/usr/bin/env python3
"""CUTLASS buffer ownership and official FA3 overlap; ordinal time, not profiling.

Source versions: CUTLASS e05f953a; FA3 060c9188. The companion tutorial
records exact source locations. Reuses the repository's SVG / PNG renderer.
Run with an optional output directory; existing diagrams are left untouched.
"""
from pathlib import Path
import argparse
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD, GOLD_LIGHT, PANEL, MUTED
import cute_r2s_retile as renderer

WIDTH = 1600
GREEN, GREEN_LIGHT = '#237457', '#E0F1E8'


def canvas(title, subtitle, height):
    d = Drawing(title, subtitle, height)
    d.parts[0] = d.parts[0].replace('width="1200"', f'width="{WIDTH}"').replace(
        f'viewBox="0 0 1200 {height}"', f'viewBox="0 0 {WIDTH} {height}"')
    d.parts[3] = d.parts[3].replace('width="1200"', f'width="{WIDTH}"')
    return d


def arrow(d, x1, y1, x2, y2, color=BLUE, dash=False):
    dashed = ' stroke-dasharray="5 4"' if dash else ''
    d.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="1.5"{dashed}/>')
    if y1 == y2:
        pts = f'{x2},{y2} {x2-7},{y2-4} {x2-7},{y2+4}'
    else:
        pts = f'{x2},{y2} {x2-4},{y2-7} {x2+4},{y2-7}'
    d.parts.append(f'<polygon points="{pts}" fill="{color}"/>')


def box(d, x, y, w, label, detail='', fill=PANEL, h=66):
    d.rect(x, y, w, h, fill)
    d.text(x+w/2, y+27 if detail else y+h/2+7, label, 20, bold=True, anchor='middle')
    if detail:
        d.text(x+w/2, y+52, detail, 17, MUTED, anchor='middle')


def lifecycle():
    d = canvas('CUTLASS Pipeline：管理缓冲槽的写入、读取与复用',
               '每个 slot 都遵守相同协议；横轴是合法先后与重叠示意，不是实测周期。', 700)
    ops = [
        (40, 220, 'producer_acquire', '等 empty：允许覆盖'),
        (300, 210, 'TMA load', '提交异步传输'),
        (550, 240, 'consumer_wait', '等 full：数据已到达'),
        (830, 210, 'MMA consume', '输入仍可能被读取'),
        (1080, 220, 'completion', '最后一次读取已结束'),
        (1340, 220, 'consumer_release', '归还 empty'),
    ]
    for (x,w,*_), nxt in zip(ops,ops[1:]):
        arrow(d,x+w,157,nxt[0]-8,157)
    for x,w,label,detail in ops:
        box(d,x,122,w,label,detail,BLUE_LIGHT if 'wait' in label or 'acquire' in label else PANEL)
    d.text(40,224,'TMA 完成时硬件更新 full barrier；PipelineTmaAsync 的 producer_commit 通常是 no-op。',21,BLUE)
    d.text(40,258,'UMMA 类 Pipeline 可把 release 注册为 MMA 完成后的通知；调用 release 返回不代表已经可覆盖。',21)

    x0, unit = 240, 135
    arrow(d,x0,306,1560,306)
    d.text(1560,332,'逻辑时间 →',17,MUTED,anchor='end')
    d.text(40,383,'TMA 硬件',22,bold=True)
    d.text(40,493,'Tensor Core',22,bold=True)
    loads=[(0,1,'load 0 / slot 0'),(1.05,2.05,'load 1 / slot 1'),
           (2.1,3.1,'load 2 / slot 2'),(3.25,4.25,'load 3 / slot 0'),
           (5.2,6.2,'load 4 / slot 1')]
    compute=[(1.1,3.1,'MMA 0 / slot 0'),(3.15,5.15,'MMA 1 / slot 1'),
             (5.2,7.2,'MMA 2 / slot 2'),(7.25,9.25,'MMA 3 / slot 0')]
    for start,end,label in loads:
        top,bottom=label.split(' / ')
        box(d,x0+start*unit,346,(end-start)*unit,top,bottom,fill=BLUE_LIGHT,h=62)
    for start,end,label in compute:
        box(d,x0+start*unit,461,(end-start)*unit,label,fill=GOLD_LIGHT,h=52)
    # Completion marker up to refill is a dependency; route via the empty gap.
    d.parts.append(f'<path d="M{x0+3.1*unit},514 L{x0+3.1*unit},432 L{x0+3.25*unit},432 L{x0+3.25*unit},411" fill="none" stroke="{GREEN}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    d.text(40,561,'slot 0：load 0 → MMA 0 完成 → 才允许 load 3；满环产生背压，不能覆盖在途输入。',22,GREEN,True)
    d.text(40,605,'3-slot consumer state：(0,0) → (1,0) → (2,0) → (0,1) → (1,1) → (2,1) → (0,0)',22)
    d.text(40,648,'(index, phase) 区分位置和轮次。producer 初始 phase=1，consumer 初始 phase=0。',21,MUTED)
    return d


def fa3():
    d=canvas('FA3 / Hopper：一个 consumer 内如何把 softmax 藏进 PV 的执行窗口',
             '官方 060c9188：IntraWGOverlap=true、RescaleOBeforeGemm=false；j 是 KV 遍历序号。',720)
    box(d,40,114,445,'WG0：producer 预留组','普通 TMA 路径：1 warp 工作，3 warps 返回',BLUE_LIGHT)
    box(d,515,114,510,'WG1：consumer 0','64 行 Q；QK + softmax + PV；累加结果在寄存器',GOLD_LIGHT)
    box(d,1055,114,505,'WG2：consumer 1','另外 64 行 Q；共享 K / V，独立 S / P / O',GREEN_LIGHT)
    d.text(40,217,'普通 BF16 D128、两 consumer、TMA KV：寄存器额度 24 / 240 / 240（每线程）。',22,BLUE)
    d.text(40,253,'Prologue：QK(0) → wait<0> → softmax(0)；之后进入下面的稳态。',22)
    arrow(d,225,299,1550,299)
    d.text(1548,324,'提交与执行分开画；非实测时间',17,MUTED,anchor='end')
    d.text(40,381,'本 WG 控制 /',20,bold=True)
    d.text(40,410,'CUDA cores',20,bold=True)
    # TC executes QK then PV; software wait<1> exposes QK and overlaps softmax with PV.
    ops=[(225,145,'issue QK','QK(j+1)'),(385,135,'issue PV','PV(j)'),
         (535,145,'wait<1>','QK 已完成'),(700,325,'softmax(j+1)','读 S；计算新 scale / P'),
         (1045,145,'wait<0>','PV 已完成'),(1210,340,'rescale O','缩放旧累计结果；随后可继续')]
    for x,w,l,t in ops:box(d,x,350,w,l,t,BLUE_LIGHT if l.startswith('softmax') else PANEL)
    d.text(40,495,'Tensor Core',20,bold=True)
    box(d,225,463,455,'QK(j+1) 在途','shared Q / K → 寄存器 S',GOLD_LIGHT)
    box(d,700,463,490,'PV(j) 在途','寄存器 P + shared V → 寄存器 O',GREEN_LIGHT)
    for x,color in [(680,BLUE),(1190,GREEN)]:
        d.parts.append(f'<path d="M{x},416 L{x},530" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    d.text(225,566,'wait<1> 后释放 K(j+1)；wait<0> 后释放 V(j)。P 寄存器也不能在 PV 读完前覆盖。',22)
    d.text(40,612,'另一 consumer WG 可以在本组做 softmax 时提交 MMA；源码用 NamedBarrier 调整发射节奏。',22,BLUE)
    d.text(40,651,'producer 同时预取未来 K/V；所有 reader 按协议 release 后，该 slot 才能复用。',21)
    d.text(40,690,'Epilogue：补发最后 PV → 排空 MMA → O / rowsum → 类型转换与布局整理 → global O。',21,MUTED)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir',nargs='?',type=Path,default=Path(__file__).parent)
    p.add_argument('--svg-only',action='store_true')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,fn in [('cutlass_pipeline_lifecycle',lifecycle),('cutlass_fa3_pipeline',fa3)]:
        d=fn();f=args.output_dir/f'{name}.svg';f.write_text(d.svg(),encoding='utf-8')
        if not args.svg_only:
            renderer.W,renderer.H=WIDTH,d.height
            renderer.render_png(f,f.with_suffix('.png'))


if __name__=='__main__':
    main()
