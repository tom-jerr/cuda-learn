#!/usr/bin/env python3
"""Editable figures for FlashMLA 414a2f3 and the April 2025 TMA comparison.

All timelines are schematic. Positional argument selects the output directory.
Uses the repository's existing SVG drawing and librsvg PNG rendering helpers.
"""
import argparse
from pathlib import Path
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD, GOLD_LIGHT, PANEL, MUTED
import cute_r2s_retile as renderer

RED, RED_LIGHT = '#9e3030', '#fae5e5'

def arrow(d,x,y,xx,yy,color=BLUE):
    assert x==xx or y==yy
    d.parts.append(f'<path d="M{x},{y} L{xx},{yy}" stroke="{color}" stroke-width="2" fill="none"/>')
    if y==yy:
        s=1 if xx>x else -1
        points=f'{xx},{yy} {xx-8*s},{yy-5} {xx-8*s},{yy+5}'
    else:
        s=1 if yy>y else -1
        points=f'{xx},{yy} {xx-5},{yy-8*s} {xx+5},{yy-8*s}'
    d.parts.append(f'<polygon points="{points}" fill="{color}"/>')

def box(d,x,y,w,h,title,lines,fill=PANEL):
    d.rect(x,y,w,h,fill)
    d.text(x+15,y+29,title,21,bold=True)
    for i,line in enumerate(lines):d.text(x+15,y+59+29*i,line,18)

def fused():
    d=Drawing('Q 的 token × head 融合，以及两次 GEMM 的维度',
              '首版 BF16 dense decode；这里 h_kv=1，s_q=2，h_q=128，d_k=576，d_v=512。',890)
    box(d,40,115,500,128,'原始 Q：shape=(2,128,576)',
        ['stride=(73728,576,1) BF16 elements','地址：q×73728 + h×576 + f'],BLUE_LIGHT)
    arrow(d,550,178,625,178)
    box(d,640,115,520,128,'融合后 Q：shape=(256,576)',
        ['m=q×128+h；stride=(576,1)','地址：m×576+f；此例 reshape 无搬运'],BLUE_LIGHT)
    d.text(40,290,'融合后的 M 向下：先遍历 head，再进入下一个 query token。',22,bold=True)
    for i in range(4):
        y=325+i*64
        d.rect(100,y,410,58,BLUE_LIGHT if i<2 else GOLD_LIGHT)
        d.text(115,y+36,f'M {64*i}..{64*i+63}  =  q{i//2}, h{64*(i%2)}..{64*(i%2)+63}',19)
        d.text(580,y+36,f'CTA m_block={i}：64×576 的 Q tile',21)
    arrow(d,67,333,67,566)
    d.text(40,610,'固定一个 CTA：同一份 KV 被它的 64 个 Q 行共同使用。',22,bold=True)
    box(d,40,633,535,123,'QK：归约维是 feature',
        ['Q[64,576] × K[64,576]ᵀ → S[64,64]','WGMMA m64n64k16：576/16=36 次'],BLUE_LIGHT)
    box(d,605,633,555,123,'PV：归约维是 token',
        ['P[64,64] × V[64,512] → O[64,512]','2 个 WG 各输出64×256，各4次 k16'],GOLD_LIGHT)
    d.text(40,804,'M 超过64会增加 CTA 数；并非单 CTA 的 M 随 h_q×s_q 无限扩大。',21,BLUE)
    d.text(40,844,'跨 M tile 的 KV 复用还依赖 L2；reshape 本身不能保证每份 KV 只从 HBM 读取一次。',19,MUTED)
    return d

def scheduler():
    d=Drawing('序列维 Stream-K：先串接请求，再按带开销的预算分区',
              '可复算的首版调度例：8 SM，2 个 M tiles，1 KV head → 4 partitions → grid=(2,1,4)。',950)
    box(d,40,112,1120,125,'每请求块数 A=2，B=3，C=40，D=7；每块64 tokens',
        ['fixed_overhead=5；total=(2+3+40+7)+4×5=72',
         'payload=ceil(72/4)+5=23；每新的一段请求额外扣5'],BLUE_LIGHT)
    d.text(40,284,'序列方向 →；分割点仅落在64-token块边界。',22,bold=True)
    widths=[100,110,660,250];x=40
    for name,w,fill in zip(['A:2','B:3','C:40','D:7'],widths,[PANEL,GOLD_LIGHT,BLUE_LIGHT,PANEL]):
        d.rect(x,305,w,54,fill);d.text(x+w/2,340,name,22,anchor='middle');x+=w
    arrow(d,40,380,1150,380)
    rows=[['partition','工作区间（右端不含）','真实块数','估计成本'],
          ['p0','A[0,2) + B[0,3) + C[0,3)','8','8 + 3×5 = 23'],
          ['p1','C[3,21)','18','18 + 5 = 23'],
          ['p2','C[21,39)','18','18 + 5 = 23'],
          ['p3','C[39,40) + D[0,7)','8','8 + 2×5 = 18']]
    d.table(40,414,[155,515,160,290],rows,rh=51)
    d.text(40,714,'A、B、D：no-split，直接写 O；C：4份 partial O/LSE，随后 combine。',21,bold=True)
    d.text(40,756,'num_splits 是前缀和：[0,1,2,6,7]；C 的4份结果位于 slots [2,6)。',20,BLUE)
    box(d,40,785,1120,119,'一个 partition 对每个 m_block 都有一个 CTA',
        ['p0 对应两个 CTA，分别计算 heads 0..63 与64..127；其他分区同理。',
         'CTA 不与 SM 编号绑定；小工作量、取整、尾块和估算误差仍可能造成空闲。'])
    return d

def g2s():
    d=Drawing('G2S 平铺：16×64 copy tile → 64×576 shared tile',
              'WG1 的局部线程 u=0..127；每次 cp.async 搬16 bytes=8 BF16。所有 offset 以 BF16 计。',1080)
    box(d,40,111,545,128,'线程 layout：shape=(16,8), stride=(8,1)',
        ['u=8r₀+c₀；value shape=(1,8)',
         '每轮128线程×8值 =16行×64列'],BLUE_LIGHT)
    box(d,615,111,545,128,'线程 u 的完整搬运坐标',
        ['r=floor(u/8)+16a，a=0..3',
         'f=8(u mod 8)+v+64j，v=0..7，j=0..8'],GOLD_LIGHT)
    d.text(40,282,'每格是一个16×64 tile：纵向4份，横向9份。',22,bold=True)
    for j in range(9):d.text(135+110*j,329,f'j={j}',18,anchor='middle')
    for a in range(4):
        d.text(65,381+49*a,f'a={a}',18,anchor='end')
        for j in range(9):
            d.rect(80+j*110,351+a*49,110,49,BLUE_LIGHT if j==0 else PANEL)
            d.text(135+j*110,382+a*49,'16×64',18,anchor='middle')
    arrow(d,1120,354,1120,540)
    d.text(1110,572,'外层 a ↓',17,BLUE,anchor='middle')
    arrow(d,88,580,1058,580)
    d.text(500,612,'每个 a 内，源码沿 j → 发出9个向量 copy；v 在一条16B指令内。',18,BLUE,anchor='middle')
    d.table(40,644,[340,380,400],[
        ['逻辑移动','global（连续物理页）','shared（SW128）'],
        ['v → v+1（包内）','+1','+1'],
        ['a → a+1（向下16行）','+16×576 = +9216','+16×64 = +1024'],
        ['j → j+1（向右64列）','+64','+64×64 = +4096'],
        ['切换整个 K buffer','重新查 block_table','±64×576 = ±36864'],
    ],rh=47)
    d.text(40,923,'global per-thread: shape=((8,1),4,9), stride=((1,0),9216,64)',21,mono=True)
    d.text(40,967,'shared: W(r,f)=4096⌊f/64⌋+64r+((f mod64) XOR 8(r mod8))',20,BLUE)
    d.text(40,1010,'例 u=8：r=1、f=0 → global=576，shared=72；包内的8个值仍连续。',20)
    d.text(40,1047,'上图箭头表示逻辑遍历；异步 copy 的完成次序不能据此推断。',18,MUTED)
    return d

def swizzle():
    d=Drawing('Swizzle 推导：把行号的3位 XOR 到16-byte包编号',
              '一个8×64 BF16 atom；r=0..7，c=8g+v，g=0..7，v=0..7。',1120)
    box(d,40,112,1120,129,'元素地址 e = 64r+8g+v：二进制拆成 [ rrr | ggg | vvv ]',
        ['Sw<3,3,3>：保持 vvv，把 rrr 右移3位并 XOR 到 ggg。',
         '结果 e′=64r+8(g XOR r)+v；行内8值的顺序保持不变。'],BLUE_LIGHT)
    d.text(40,288,'表内数字=物理包编号 g XOR r；列是逻辑包 g。每包16 bytes。',22,bold=True)
    widths=[160]+[120]*8
    rows=[['r / g']+[str(g) for g in range(8)]]+[[f'r={r}']+[str(g^r) for g in range(8)] for r in range(8)]
    d.table(40,315,widths,rows,rh=42)
    for r in range(8):
        d.rect(200,357+42*r,120,42,BLUE_LIGHT)
        d.text(260,385+42*r,str(r),20,BLUE,True,anchor='middle')
    d.text(40,737,'固定逻辑 g=0，向下读8行：物理包依次为0,1,2,3,4,5,6,7。',21,BLUE,True)
    d.table(40,766,[280,420,420],[
        ['同一128B访问组','没有 swizzle','有 swizzle'],
        ['8行各取一个16B包','都落在 banks 0..3','分别落在0..3、4..7、…、28..31'],
        ['总大小 8×16B=128B','相同bank、不同地址：冲突','覆盖32 banks：该访问模式无冲突'],
    ],rh=46)
    box(d,40,940,1120,130,'改用 byte 地址：a=2e → [ rrr | ggg | vvv | 0 ]',
        ['源位由[8:6]变为[9:7]；目标位由[5:3]变为[6:4]；距离仍为3。',
         'B=3 不变，S=3 不变，M=3+1=4 → Sw<3,4,3>。'],GOLD_LIGHT)
    return d

def storage():
    d=Drawing('Shared layout：8×64 atom 怎样平铺成64×576',
              'K/Q 使用 SW128。图中每格是8行×64列的 atom；格内先按列连续，再应用 XOR。',1050)
    box(d,40,111,1120,125,'先不考虑 swizzle：atom shape=(8,64), stride=(64,1)',
        ['r=r₀+8r₁，f=f₀+64j；shape=((8,8),(64,9))',
         'stride=((64,512),(1,4096))；e=64r₀+512r₁+f₀+4096j'],BLUE_LIGHT)
    for j in range(9):d.text(149+108*j,287,f'j={j}',17,anchor='middle')
    for r1 in range(8):
        d.text(83,330+39*r1,f'r₁={r1}',17,anchor='end')
        for j in range(9):
            d.rect(95+j*108,303+r1*39,108,39,BLUE_LIGHT if j<8 else GOLD_LIGHT)
            d.text(149+j*108,328+39*r1,str(512*r1+4096*j),16,anchor='middle')
    arrow(d,1100,309,1100,611)
    d.text(1125,448,'↓',22,BLUE)
    d.text(1125,480,'+512',16,BLUE,anchor='middle')
    arrow(d,98,649,1055,649)
    d.text(550,680,'先沿 r₁ ↓ 放8个 atom，组成64×64条带；再沿 j →，每次 +4096。',19,BLUE,anchor='middle')
    d.text(40,723,'格中数字是 BF16 元素 offset；蓝色前8条带同时充当 V，黄色最后条带只用于 QK。',20)
    box(d,40,755,545,178,'K 的逻辑坐标：(token,feature)',
        ['shape=(64,576)；共9个64×64条带',
         'V 使用 feature0..511 的前8条带。',
         '一个K槽：36864 BF16=72 KiB；',
         '两个K槽：144 KiB。'],BLUE_LIGHT)
    box(d,615,755,545,178,'PV 的 B view：(feature,token)',
        ['shape=(512,64)；Vt(f,r)=K(r,f)',
         '底层 stride=((1,4096),64)',
         '仍指向相同 shared bytes；',
         '改变 view 不等于搬运/转置数据。'],GOLD_LIGHT)
    d.text(40,987,'Swizzle只重排格内16B包；外层条带的 +512、+4096 仍成立。',21,bold=True)
    d.text(40,1023,'同一逻辑 K(1,0) 与 Vt(0,1) 都在元素地址72；完整公式见 Swizzle 图。',19,MUTED)
    return d

def registers():
    d=Drawing('WGMMA 的结果分布、softmax 行视图与 P 的交接',
              'QK 由一个128线程 warpgroup 执行；图中的 lane 所有权描述结果寄存器，不描述 shared 读指令。',1080)
    box(d,40,112,1120,131,'u=32w+lane；线程内 slot v=0..31',
        ['row=16w+floor(lane/4)+8×floor((v mod4)/2)',
         'col=2(lane mod4)+8×floor(v/4)+(v mod2)'],BLUE_LIGHT)
    d.text(40,287,'warp0 / lane0：最前8个 slot，按 v=0→7 的顺序展开。',22,bold=True)
    rows=[['slot v','0','1','2','3','4','5','6','7'],
          ['(row,col)','(0,0)','(0,1)','(8,0)','(8,1)','(0,8)','(0,9)','(8,8)','(8,9)']]
    d.table(40,312,[160]+[120]*8,rows,rh=52)
    arrow(d,213,450,1120,450)
    d.text(40,490,'相邻列 → 下一行(+8) → 下一组列(+8)；之后重复，直到 token63。',21,BLUE)
    d.table(40,525,[340,390,390],[
        ['view','shape','stride（线程内元素槽）'],
        ['QK accumulator','((2,2,8),1,1)','((1,2,4),0,0)'],
        ['softmax row / col','((2,1),(2,8,1))','((2,0),(1,4,0))'],
        ['P → RS A 的重分组','((2,2,2),1,4)','((1,2,4),0,8)'],
    ],rh=49)
    d.text(40,763,'softmax 每线程2行×16列；同一行跨 lane0..3，共同归约64列。',21,bold=True)
    d.text(40,803,'归约使用4-lane shuffle；P 重分组只改变 view，不做线程间搬运。',20)
    box(d,40,835,1120,180,'首版跨 WG 的 sP：按 fragment 槽位暂存，不使用 SW128 P 布局',
        ['shape=((2,2),128,1,8)，stride=((1,2),4,0,512)',
         'addr_sP(u,v)=(v mod4)+4u+512×floor(v/4)',
         'WG0 lane u 写出 → SReady barrier → WG1 lane u 读回 → 两边用 RS PV。'],GOLD_LIGHT)
    d.text(40,1052,'每个WG保留64×256个FP32输出=128值/线程；两WG合计32768个FP32累加值。',18,MUTED)
    return d

def mask():
    d=Drawing('Mask 与逆序：把不规则尾部放在最前面处理',
              '例 L=130，s_q=4，ngroups=16；融合 M=64。每行代表同一 query 的16个 heads。',1040)
    d.text(40,126,'可见条件：k ≤ L−s_q+q = 126+q。横向只放大 tokens 120..135。',22,bold=True)
    for j in range(16):d.text(240+56*j+28,181,str(120+j),17,anchor='middle')
    for q in range(4):
        d.text(210,230+62*q,f'q{q} / M{16*q}..{16*q+15}',18,anchor='end')
        for j in range(16):
            k=120+j;fill=BLUE_LIGHT if k<=126+q else RED_LIGHT if k>=130 else GOLD_LIGHT
            d.rect(240+56*j,198+q*62,56,62,fill)
            d.text(268+56*j,237+62*q,'✓' if k<=126+q else 'OOB' if k>=130 else 'mask',15,anchor='middle')
    d.text(40,488,'蓝：有效；黄：真实cache token，但属于未来；红：超出当前 L 的未初始化位置。',20)
    arrow(d,240,520,679,520);arrow(d,692,520,1129,520)
    d.text(461,551,'block1 的尾部（k120..127）',19,anchor='middle')
    d.text(908,551,'block2（k128..191）的前8列',19,anchor='middle')
    box(d,40,590,350,125,'① block2 → buffer0',
        ['先算；尾块copy按 L 清零', 'S 还需逐行 causal mask'],RED_LIGHT)
    box(d,425,590,350,125,'② block1 → buffer1',
        ['全量copy；Q0仍需mask', '其他query可见整块'],GOLD_LIGHT)
    box(d,810,590,350,125,'③ block0 → buffer0',
        ['全量copy', '无需逐元素mask'],BLUE_LIGHT)
    arrow(d,390,650,422,650);arrow(d,775,650,807,650)
    d.text(40,765,'首版 n_masking_steps：非causal=1；causal=ceil(64/64)+1=2。',22,BLUE,True)
    d.text(40,807,'先处理右边，再向左走；后面的完整块可用 Is_even_MN=true copy 快路径。',21)
    box(d,40,840,1120,149,'TMA 的 tensor 边界 ≠ 每个请求的有效长度',
        ['block2 实际物理page含64个槽，TMA看来全部在范围内；只有前2槽属于本请求。',
         '首版：无效行不copy，cute::clear(shared)；4月版：TMA完成后 fill_oob_V(shared)。',
         'S 的 mask 解决概率；V 的清零解决 0×NaN=NaN。两个步骤缺一不可。'])
    return d

def pipeline():
    d=Drawing('首版双缓冲：WG1 发起搬运，WG0 同时计算 QK / softmax',
              '示意时序，不按实际周期比例。i=2→1→0；偶数block用buffer0，奇数block用buffer1。',900)
    arrow(d,205,137,1154,137)
    d.text(1120,122,'时间 →',18,BLUE,anchor='end')
    ys=[205,315,425];names=['WG0','WG1','异步 copy']
    for y,name in zip(ys,names):
        d.text(40,y+31,name,21,bold=True)
        d.rect(200,y,950,66,PANEL)
    def event(x,y,w,label,fill):
        d.rect(x,y,w,66,fill);d.text(x+w/2,y+40,label,17,anchor='middle')
    event(210,205,145,'QK₂',BLUE_LIGHT);event(355,205,105,'softmax₂',GOLD_LIGHT)
    event(480,205,130,'PV₂ 左',BLUE_LIGHT)
    event(210,315,85,'issue K₁',GOLD_LIGHT);event(295,315,165,'等 SReady',PANEL)
    event(460,315,55,'读P',GOLD_LIGHT);event(515,315,95,'PV₂右',BLUE_LIGHT)
    event(295,425,330,'K₁→buffer1 在途',GOLD_LIGHT)
    d.parts.append('<path d="M650,180 L650,510" stroke="#5D687A" stroke-dasharray="5 5"/>')
    arrow(d,460,275,460,310)
    d.text(454,296,'SReady',14,BLUE,anchor='end')
    event(670,205,140,'QK₁',BLUE_LIGHT);event(810,205,100,'softmax₁',GOLD_LIGHT)
    event(930,205,130,'PV₁ 左',BLUE_LIGHT)
    event(670,315,85,'issue K₀',GOLD_LIGHT);event(755,315,155,'等 SReady',PANEL)
    event(910,315,55,'读P',GOLD_LIGHT);event(965,315,95,'PV₁右',BLUE_LIGHT)
    event(755,425,335,'K₀→buffer0 在途',GOLD_LIGHT)
    arrow(d,910,275,910,310)
    d.text(904,296,'SReady',14,BLUE,anchor='end')
    d.text(650,546,'CTA barrier：当前PV都完成；下一块copy也已wait完成',19,anchor='middle')
    d.text(40,595,'第一次迭代之前：WG1 先搬 Q 与 K₂，cp_async_wait<0>() 后全CTA同步。',21)
    box(d,40,625,1120,181,'缓冲槽复用的必要条件',
        ['buffer0 中的 K₂ 同时充当 V₂；WG0/WG1 的 PV₂ 都读完，才可覆盖为 K₀。',
         'cp_async_wait<0>() 等搬运完成；__syncthreads() 交接数据并隔开迭代。',
         'WG1 不是纯 producer：发起下一块copy之后，还要做本块的右半 PV。',
         '首版每次 gemm 使用 warpgroup_wait<0>()；softmax 与本WG的QK不会重叠。'])
    d.text(40,855,'4月版改为两WG交替QK/softmax/PV，并按64 feature条带发起TMA；不能照搬本图。',19,MUTED)
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir',nargs='?',type=Path,default=Path(__file__).parent)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for suffix,fn in [('fused',fused),('scheduler',scheduler),('g2s',g2s),('storage',storage),('swizzle',swizzle),
                      ('registers',registers),('mask',mask),('pipeline',pipeline)]:
        d=fn();path=args.output_dir/f'flashmla_initial_{suffix}.svg'
        path.write_text(d.svg(),encoding='utf-8')
        renderer.W,renderer.H=1200,d.height
        renderer.render_png(path,path.with_suffix('.png'))
        print(path)

if __name__=='__main__':main()
