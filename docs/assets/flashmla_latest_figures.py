#!/usr/bin/env python3
"""FlashMLA 15f13e5: editable, source-derived diagrams; no measured timing.
Usage: python3 docs/assets/flashmla_latest_figures.py [output_dir]
"""
import argparse
from pathlib import Path
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD, GOLD_LIGHT, PANEL, MUTED
from flashmla_initial_figures import arrow, box
import cute_r2s_retile as renderer


def layouts():
    d = Drawing('先排列 atom，再读 stride：dense SW128 与 sparse INTER',
                'Hopper，64 tokens × 576 features；地址差单位 BF16。箭头表示存储平铺顺序。', 1070)
    for x, name, atom, strip, stride, fill in [
        (40, 'Dense：SW128', '8×64 atom = 512', '64×64 条带 = 4096', '+4096', BLUE_LIGHT),
        (650, 'Sparse：INTER', '8×8 atom = 64', '64×8 条带 = 512', '+512', GOLD_LIGHT)]:
        d.text(x, 135, name, 24, bold=True)
        d.text(x, 169, atom+' BF16', 19)
        for c in range(3):
            for r in range(8):
                d.rect(x+30+c*128, 198+r*36, 128, 36, fill if c == 0 else PANEL)
                d.text(x+94+c*128, 222+r*36, f'A{r+8*c}', 17, anchor='middle')
        arrow(d, x+12, 201, x+12, 482)
        d.text(x+30, 515, '① 先沿行向下放满8个 atom', 19, BLUE)
        arrow(d, x+35, 543, x+410, 543)
        d.text(x+30, 577, '② 再向右一条带：'+stride, 19, BLUE)
        d.text(x+30, 615, strip+' BF16', 20, bold=True)
    d.table(40, 653, [260,430,430], [
        ['同一个逻辑移动', 'Dense SW128', 'Sparse INTER'],
        ['向下8 token行', '+512', '+64'],
        ['向右8 features', '包位置受 XOR 影响', '+512：跨过整个64×8条带'],
        ['向右64 features', '+4096：1条带', '+4096：8条小条带'],
        ['向右256 features', '+16384', '+16384'],
    ], rh=47)
    d.text(40, 935, 'Dense 条带内部：16B包编号 g → g XOR (r mod8)；8个 BF16 包内顺序不变。', 20)
    d.text(40, 979, 'Sparse 一个16-FP8加载解出16个BF16：拆为两次8-BF16写入，目的地址相差512。', 20)
    d.text(40, 1026, '两者的 V 都是 K 前512维的转置 view；地址复用，并没有另做 shared transpose。', 20, BLUE)
    return d


def dense_alias():
    d = Drawing('Dense：用 Q 的最后一条带存 P1',
                'M=64，Dqk=576=9×64；每个条带64×64 BF16 = 8 KiB。', 880)
    d.text(40, 133, '① Q 经 TMA 到 shared：9条带，72 KiB', 23, bold=True)
    for j in range(9):
        d.rect(40+j*124, 160, 124, 80, GOLD_LIGHT if j==8 else BLUE_LIGHT)
        d.text(102+j*124, 192, f'Q{j}', 22, anchor='middle')
        d.text(102+j*124, 220, f'{j*64}..{j*64+63}', 15, anchor='middle')
    arrow(d, 1094, 245, 1094, 287)
    box(d, 650, 295, 506, 118, '② 两个 WG 各保存一份 rQ8',
        ['ldmatrix.x4 → 每线程32 BF16', 'shape=((2,2,2),1,4)，即16个32-bit槽'], GOLD_LIGHT)
    box(d, 40, 295, 575, 118, '存活条件',
        ['WG0 首块完整 SS QK 已读完 Q8，', '两个 WG 的 rQ8 都已就绪，才可覆盖 Q8。'])
    arrow(d, 1094, 418, 1094, 461)
    d.text(40, 472, '③ 稳态：Q0..Q7 留在 shared；原 Q8 空间变为 sP1', 23, bold=True)
    for j in range(9):
        d.rect(40+j*124, 499, 124, 75, GOLD_LIGHT if j==8 else BLUE_LIGHT)
        d.text(102+j*124, 544, 'P1' if j==8 else f'Q{j}', 22, anchor='middle')
    d.table(40, 612, [300,420,400], [
        ['MMA 工作', 'A 操作数来源', 'B 操作数来源'],
        ['QK 的 features 0..511', 'sQ：SS WGMMA，32次 k16', 'sK 对应8条带'],
        ['QK 的 features 512..575', 'rQ8：RS WGMMA，4次 k16', 'sK 第9条带'],
        ['WG0 计算 P1 × V1L', 'sP1：SS WGMMA', 'sK1 的 V 左半'],
    ], rh=46)
    d.text(40, 837, '节省8 KiB shared，代价是每个WG持有 rQ8；它同时改变了稳态 QK 的指令类型。', 20, BLUE)
    return d


def dense_wait():
    d = Drawing('Dense：wait<N> 留住较新的 QK，让旧 V 缓冲先释放',
                '按源码发射顺序画事件；横轴为先后关系，方框宽度不代表时钟或执行时间。', 960)
    d.text(40, 132, 'WG0：同一个 WGMMA group 队列', 23, bold=True)
    events=[('① PV1L', '旧 V1L 仍在读'), ('② QK2[0]', 'commit group'),
            ('③ QK2[1]', 'commit group'), ('④ QK2[2]', 'commit group'), ('⑤ QK2[3]', 'commit group')]
    for i,(a,b) in enumerate(events):
        x=40+i*224
        box(d,x,160,210,96,a,[b],GOLD_LIGHT if i==0 else BLUE_LIGHT)
        if i<4: arrow(d,x+212,207,x+223,207)
    arrow(d, 40, 285, 1150, 285)
    d.text(40, 327, '⑥ wait<4>：最多保留4个较新 group 未完成 → 更早的 PV1L 必须完成', 22, bold=True)
    box(d,40,353,540,120,'旧 V1L 被释放', ['TMA 开始写入 K3 的左4条带', '该槽位：V1L → K3 features 0..255'], GOLD_LIGHT)
    box(d,620,353,540,120,'较新的 QK2 可仍在执行', ['继续等待/发射 QK2 的后5条带', '最后 wait<0>，才能对 rP2 做 softmax'], BLUE_LIGHT)
    d.text(40, 527, 'WG1：把两个 PV 的完成点拆开，分别释放右半缓冲', 23, bold=True)
    d.table(40,553,[135,450,535],[
        ['顺序','WGMMA / wait','允许开始的 TMA'],
        ['①','发射 PV1R，然后发射 PV0R','此时不能据“已发射”就覆盖输入'],
        ['②','wait<1>：PV1R 完成','K3 → K1 槽位的右半 + RoPE'],
        ['③','wait<0>：PV0R 也完成','K2 → K0 槽位的右半 + RoPE'],
        ['④','QK3 按 4,5,6,7,8,0,1,2,3 发射','逐条带等待各自 TMA barrier'],
    ],rh=47)
    d.text(40,835,'一个64-feature QK条带：4条 k16 WGMMA → 1个 commit group。',21,BLUE)
    d.text(40,881,'wait<4> 数的是提交组，不是4条WGMMA；它也不保证这4组一定仍未完成。',20)
    d.text(40,924,'“Issued” named barrier 与“输入不再被读”的完成条件不同，不能互换。',19,MUTED)
    return d


def sparse_cluster():
    d=Drawing('Hopper sparse FP8：两 CTA 分摊加载，交换 BF16 KV',
              'V32，128 query heads，cluster=2；一个计算块含64个被选中的 token。',1070)
    for x,cta,heads,rows in [(40,0,'0..63','0..31'),(650,1,'64..127','32..63')]:
        box(d,x,115,510,117,f'CTA{cta}：Q heads {heads}',
            [f'WG2 gather + dequant：selected rows {rows}', '每CTA 384线程：WG0、WG1、WG2'],BLUE_LIGHT if cta==0 else GOLD_LIGHT)
        box(d,x,278,510,116,'自己的 shared K：完整64×576 BF16',
            ['32行本地产生，32行由对端写入', '每个CTA仍占72 KiB；两槽共144 KiB'])
        box(d,x,485,510,117,'WG0 / WG1 计算各自的64个 heads',
            ['WG0：QK + softmax + PV 左256维', 'WG1：接收 P/scale，PV 右256维'])
        arrow(d,x+250,237,x+250,273)
        arrow(d,x+250,399,x+250,480)
    arrow(d,552,317,647,317)
    arrow(d,648,356,553,356,GOLD)
    d.text(600,427,'每方向 32×576×2 = 36 KiB；st.async 写 DSM',19,anchor='middle',color=BLUE)
    d.text(40,652,'一个缓冲槽的生命周期（两槽交替复用）',23,bold=True)
    d.table(40,678,[220,460,440],[
        ['阶段','等待 / 发布事件','被证明的事实'],
        ['① 可覆盖','两CTA的4个consumer WG全部释放','上一轮 QK/PV 不再读取此槽'],
        ['② 写入','本地普通写 + 对端 st.async','每producer只处理半个计算块'],
        ['③ 数据就绪','local_ready 与 remote_ready 都完成','本地32行与远端32行均已可读'],
        ['④ 消费','WG0 QK→P；两个WG各算半个PV','两个CTA使用同一批KV，处理不同heads'],
        ['⑤ 释放','各consumer在自己的PV完成后通知','下一轮回到①；不是仅等QK完成'],
    ],rh=46)
    d.text(40,999,'省下的是重复 gather 与反量化；MMA 仍覆盖全部128 heads，shared KV 仍每CTA一份。',20,BLUE)
    d.text(40,1040,'64 heads → cluster=1：一个producer WG分两轮处理64 tokens，不需要DSM交换。',19,MUTED)
    return d


def dual_gemm():
    d=Drawing('Blackwell head64：把 feature 对折，做两份部分 QK',
              'V32 NoPE 的512维示意；方框编号是64-feature条带，不是token block编号。',1000)
    d.text(40,133,'① 普通逻辑视图：64行 ×512 features',23,bold=True)
    for j in range(8):
        d.rect(40+j*140,158,140,63,BLUE_LIGHT if j%2==0 else GOLD_LIGHT)
        d.text(110+j*140,197,str(j),23,anchor='middle')
    d.text(40,263,'② 将相邻两个64-feature条带上下叠放：128行 ×256 features',23,bold=True)
    for row in range(2):
        for col in range(4):
            d.rect(40+col*140,288+row*68,140,68,BLUE_LIGHT if row==0 else GOLD_LIGHT)
            d.text(110+col*140,331+row*68,str(col*2+row),23,anchor='middle')
    box(d,655,288,505,136,'同一批数据，重解释为两份归约',
        ['上半：features组 0,2,4,6', '下半：features组 1,3,5,7', 'Q到TMEM；K留在shared'],PANEL)
    d.text(40,468,'③ 两份 partial scores：S_even 与 S_odd；相加后才做 softmax',23,bold=True)
    box(d,40,493,540,108,'S_even = Q_even × K_evenᵀ', ['64 query heads ×64 selected tokens'],BLUE_LIGHT)
    box(d,620,493,540,108,'S_odd = Q_odd × K_oddᵀ', ['64 query heads ×64 selected tokens'],GOLD_LIGHT)
    arrow(d,310,606,310,653)
    arrow(d,890,606,890,653)
    d.text(600,679,'S = S_even + S_odd → mask → softmax → P',24,BLUE,True,anchor='middle')
    d.table(40,720,[300,390,430],[
        ['softmax前的有效64×64 S','selected tokens 0..31','selected tokens 32..63'],
        ['heads 0..31','Warp0','Warp2'],
        ['heads 32..63','Warp1','Warp3'],
    ],rh=48)
    d.text(40,910,'Warp0↔Warp2、Warp1↔Warp3 交换 partial scores，并求和；随后交换行max与L。',20)
    d.text(40,954,'模板中 B_TOPK*2=128 表达 dual GEMM；实际仍只选择64 tokens，不是多读一倍KV。',19,MUTED)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir',nargs='?',type=Path,default=Path(__file__).parent)
    out=p.parse_args().output_dir
    out.mkdir(parents=True,exist_ok=True)
    for name,fn in [('layouts',layouts),('dense_alias',dense_alias),('dense_wait',dense_wait),('sparse_cluster',sparse_cluster),('dual_gemm',dual_gemm)]:
        d=fn(); path=out/f'flashmla_latest_{name}.svg'
        path.write_text(d.svg(),encoding='utf8')
        renderer.W,renderer.H=1200,d.height
        renderer.render_png(path,path.with_suffix('.png'))


if __name__=='__main__': main()
