#!/usr/bin/env python3
"""Editable source-derived FA4 figures (ce088ab, BF16 D128).
Schedule x positions are ordinal, never measured cycles. Optional output directory.
"""
import argparse
from pathlib import Path
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD, GOLD_LIGHT, PANEL, MUTED
from flashmla_initial_figures import arrow, box
from flashmla_pipeline_focus import wide, Timeline, GREEN, GREEN_LIGHT
import cute_r2s_retile as renderer


def roles():
    d=Drawing('FA4：每 CTA 16 warps；2-CTA cluster 共 8 个 WG',
              '固定 ce088ab 的通用 BF16 D128 路径。WG = 连续4个warp = 128线程；不是MMA发射单位。',1050)
    d.text(40,129,'Forward：q_stage=2；每 CTA 两块 Q，各128行',23,bold=True)
    d.table(40,150,[130,150,230,610],[
        ['分组','warp ID','职责','持有 / 执行'],
        ['WG0','0–3','Softmax 0','S0 → 寄存器 → rowmax / exp / rowsum → P0'],
        ['WG1','4–7','Softmax 1','S1 → 寄存器 → rowmax / exp / rowsum → P1'],
        ['WG2','8–11','Correction','O0 / O1 条件缩放；最终归一化、写 shared'],
        ['WG3','12','MMA','仅 leader CTA 的 elected thread 发射 UMMA'],
        ['WG3','13','Epilogue','shared O → global O：TMA store'],
        ['WG3','14','Load','TMA Q / K / V；等待缓冲槽可以复用'],
        ['WG3','15','Idle / scheduler','动态持久化调度启用时承担 scheduler'],
    ],rh=48)
    d.text(40,588,'Backward：固定一块 KV，沿 query tiles 循环',23,bold=True)
    d.table(40,610,[130,150,230,610],[
        ['分组','warp ID','职责','持有 / 执行'],
        ['WG0','0–3','dQ reduce','TMEM dQ → 寄存器 → shared → global reduction'],
        ['WG1+WG2','4–11','Compute','256线程协作：重建 P；计算 dS；准备 DSM 交换'],
        ['WG3','12','MMA','leader 发射五类 MMA；等待各结果 / 交换就绪'],
        ['WG3','13','Load','预取 Q / dO / LSE / D；装入 KV 相关视图'],
        ['WG3','14','Relay','等本 CTA 收到 peer dS；向 leader 汇报到达'],
        ['WG3','15','Idle','保留线程分组；不是另一个 compute WG'],
    ],rh=48)
    d.text(40,998,'2-CTA 时两边都有这些线程角色；一个联合 MMA 的发射者仍只有 leader 中的一个线程。',20,BLUE)
    return d


def operands():
    d=Drawing('Forward 2-CTA：Q 切行，K 切 token，V 切 feature',
              '一个 Q stage：联合 MMA 为256×128；每 CTA 保存128行 S / O。图为逻辑所有权，不是物理地址。',1190)
    d.text(40,132,'QKᵀ：A = Q[256,128]；B = Kᵀ[128,128]；C = S[256,128]',23,bold=True)
    d.table(40,155,[185,455,480],[
        ['逻辑对象','CTA 0 的 SMEM / TMEM','CTA 1 的 SMEM / TMEM'],
        ['Q：A 沿 M 切','query rows 0..127 × 全部128 features','query rows 128..255 × 全部128 features'],
        ['K：B 沿 N 切','KV tokens 0..63 × 全部128 features','KV tokens 64..127 × 全部128 features'],
        ['S：C 沿 M 切','128 query rows × 全部128 KV tokens','128 query rows × 全部128 KV tokens'],
    ],rh=52)
    d.text(40,409,'所以每个 softmax 行都拥有完整的128列；不需要跨 CTA 求 rowmax / rowsum。',21,BLUE)
    d.text(40,469,'PV：A = P[256,128]；B = V[128,128]；C = O[256,128]',23,bold=True)
    d.table(40,492,[185,455,480],[
        ['逻辑对象','CTA 0','CTA 1'],
        ['P：TMEM A','query rows 0..127 × 全部128 KV tokens','query rows 128..255 × 全部128 KV tokens'],
        ['V：SMEM B','全部128 KV tokens × features 0..63','全部128 KV tokens × features 64..127'],
        ['O：TMEM C','128 query rows × 全部128 features','128 query rows × 全部128 features'],
    ],rh=52)
    d.text(40,747,'tcgen05.mma.cta_group::2 由硬件组合两侧 operands；无需软件交换完整 K / V。',21,BLUE)
    d.text(40,812,'两个 Q stages 的行归属（一个 cluster work tile 共512行）',22,bold=True)
    for stage in range(2):
        y=840+stage*72
        d.text(40,y+38,f'stage {stage}',20,bold=True)
        for cta in range(2):
            r=stage*256+cta*128
            d.rect(235+cta*440,y,440,60,BLUE_LIGHT if cta==0 else GOLD_LIGHT)
            d.text(455+cta*440,y+37,f'CTA {cta}：Q rows {r}..{r+127}',21,anchor='middle')
    d.text(40,1026,'两个 stage 是两组不同的 Q 行，各有自己的 P、O 和 softmax 状态。',22,bold=True)
    d.text(40,1074,'它们复用同一批 K / V；不是把同一个 softmax 行拆成两半再分别归一化。',21)
    d.text(40,1122,'1-CTA：每 CTA 仍为2×128行；K / V 各存完整128×128，KV ring 从6槽变3槽。',20,MUTED)
    return d


def layout():
    d=Drawing('TMA 排布与 TMEM 复用：先看 tile，再读 stride',
              'BF16 D128，2-CTA。shared layout 来自本地 CuTe-DSL 4.4.2 编译期探针；无 GPU kernel 执行。',1210)
    d.text(40,128,'K：每 CTA 64 tokens ×128 features；分成两个64×64条带',22,bold=True)
    for j in range(2):
        d.rect(220+j*430,151,430,65,BLUE_LIGHT if j==0 else GOLD_LIGHT)
        d.text(435+j*430,179,f'features {j*64}..{j*64+63}',20,anchor='middle')
        d.text(435+j*430,204,f'条带起点 {j*4096} BF16',18,anchor='middle')
    arrow(d,435,237,865,237)
    d.text(650,266,'右移64 features：跨64×64 → +4096',20,BLUE,anchor='middle')
    d.table(40,297,[150,560,410],[
        ['K 的 mode','shape = ((64,16),1,(4,2),6)','stride = ((64,1),0,(16,4096),8192)'],
        ['atom 内','64 token rows；16 feature values','下移1 token +64；右移1 feature +1'],
        ['atom 外','4个K16片组成64宽；再排2个条带','右移16 features +16；换条带 +4096'],
        ['ring stage','6个槽；每槽64×128 BF16','换槽 +8192 values =16 KiB'],
    ],rh=46)
    d.text(40,511,'V 的 MMA view 是 (feature, token)：连续64 features；沿 token 接8个16-token片。',20,bold=True)
    d.text(40,552,'V shape = ((64,16),1,8,6)',22,mono=True)
    d.text(40,594,'V stride = ((1,64),0,1024,8192)',22,mono=True)
    d.text(40,635,'向下16 tokens：跨16×64 → +1024；换 stage 仍 +8192 BF16。',21,BLUE)
    d.text(40,681,'两者均叠加 SW128：物理 byte 地址 a′ = a XOR ((a & 0x380) >> 3)。',20)
    d.text(40,720,'上面的 stride 是 XOR 之前的 element layout；swizzle 不改变哪个 CTA 拥有哪些数据。',19,MUTED)
    d.text(40,786,'每 CTA 的 TMEM：128 datapaths ×512 columns ×4B =256 KiB',23,bold=True)
    x=150;unit=1.85
    for lo,hi,label,fill in [(0,128,'S0 FP32',BLUE_LIGHT),(128,256,'S1 FP32',GOLD_LIGHT),(256,384,'O0 FP32',GREEN_LIGHT),(384,512,'O1 FP32',GREEN_LIGHT)]:
        d.rect(x+lo*unit,817,(hi-lo)*unit,62,fill)
        d.text(x+(lo+hi)/2*unit,854,label,20,anchor='middle')
        d.text(x+lo*unit,806,str(lo),16)
    d.text(x+512*unit,806,'512',16,anchor='end')
    for lo,label,fill in [(64,'P0 BF16',BLUE_LIGHT),(192,'P1 BF16',GOLD_LIGHT)]:
        d.rect(x+lo*unit,907,64*unit,58,fill)
        d.text(x+(lo+32)*unit,941,label,17,anchor='middle')
        arrow(d,x+(lo+32)*unit,882,x+(lo+32)*unit,902)
    d.text(40,1008,'P 的128个BF16值只占64个32-bit列，因此覆盖对应 S 区的后半段。',21,bold=True)
    d.text(40,1055,'必须先读完 S，才可覆盖成 P；PV 消费完 P 后，下一轮 QK 才可覆盖同一 S 区。',20)
    d.text(40,1102,'Q/K/V：global → TMA → swizzled SMEM → SS/TS MMA；P 不经过 shared。',21,BLUE)
    d.text(40,1149,'普通 attention 的 K、V 是不同数据；共用 SMEM ring 是时间复用，不是 MLA 的 K=V 前缀复用。',19,MUTED)
    return d


def forward():
    d=wide('Forward：两块 Q 的 ping-pong 与独立 correction',
           '满足当前源码依赖的一种时序；非测量、不按比例。Ssj=QsKjᵀ，Osj=PsjVj；j为遍历序号。',1190)
    t=Timeline(d,x=245,step=90);t.axis(119)
    for y,n in [(170,'Load / TMA issue'),(252,'TMA 在途'),(360,'MMA issue'),(449,'Tensor Core 在途'),(550,'WG0 softmax'),(644,'WG1 softmax'),(743,'WG2 correction')]:t.lane(y,n)
    for at,lab in [(0,'K0,Q0,Q1'),(1.7,'V0'),(3.8,'K1,V1'),(8,'K2,V2')]:t.mark(170,at,lab)
    for a,b,lab in [(0,2,'K0 + Q'),(2,3.8,'V0'),(3.8,6.8,'K1,V1'),(8,10.7,'K2,V2')]:t.bar(252,a,b,lab)
    ops=[(2,3,'S00'),(3,4,'S10'),(6,7,'O00'),(7,8,'S01'),(8,9,'O10'),(9,10,'S11'),(11,12,'O01'),(12,13,'S02'),(13,14,'O11'),(14,15,'S12')]
    for a,b,lab in ops:t.mark(360,a,lab);t.bar(449,a,b,lab)
    for a,b,lab in [(3,6,'SM00 → P00'),(8,11,'SM01 → P01'),(13,16,'SM02 → P02')]:t.bar(550,a,b,lab,BLUE_LIGHT)
    for a,b,lab in [(4,7,'SM10 → P10'),(10,13,'SM11 → P11')]:t.bar(644,a,b,lab,GOLD_LIGHT)
    t.bar(743,8.5,10,'α01 O0',GREEN_LIGHT);t.bar(743,10.5,12,'α11 O1',GREEN_LIGHT)
    # These gates are exact dependencies of the schematic, not duration predictions.
    for at,y1,y2 in [(3,487,550),(4,487,644),(8,487,550),(10,487,644),(11,360,781),(13,360,682)]:t.gate(at,y1,y2)
    d.text(40,858,'PV 开始条件：V ready ∧ P 前96列 ready ∧ 对应旧 O 已 rescale（两个 CTA 都参与）。',23,bold=True)
    d.text(40,905,'展开一个 PV：发射 K=0..95 的6条 k16 MMA → 等 P 尾部 → 再发射 K=96..127 的2条。',22,BLUE)
    d.text(40,952,'当前实现先 exp/convert 整行，再分段写 TMEM；上图省略了每个 P 的96/32细分。',21)
    d.text(40,999,'Kj 等两块 Q 的 QK 完成才释放；Vj 等两块 Q 的 PV 完成才释放。异步发射返回不等于完成。',21)
    d.text(40,1046,'Prologue：S00 → S10。稳态发射：O0j → S0,j+1 → O1j → S1,j+1。',22,bold=True)
    d.text(40,1093,'Epilogue：最后 PV0 / PV1 → O 完成通知 → correction 除以 l → shared → TMA store。',21)
    d.text(40,1140,'版本差异：ce088ab 的 s0_s1_barrier=False；图中允许两组 softmax 重叠，不画论文中的 exp 互斥锁。',20,MUTED)
    return d


def exchange():
    d=Drawing('Backward 2-CTA：把 dS 从“各持半个归约”变成“各持半组输出行”',
              'query tile =128；每 CTA 原有128个KV tokens；cluster 合计256个KV tokens；dS 为 BF16。',1110)
    d.text(40,130,'计算所得：dSᵀ 按 KV 所有权分布；下面统一画成 dS[query, KV]',21,bold=True)
    d.text(425,177,'KV 0..127（CTA 0）',21,anchor='middle')
    d.text(945,177,'KV 128..255（CTA 1）',21,anchor='middle')
    for row,qlab in enumerate(['q0..63','q64..127']):
        d.text(40,246+row*102,qlab,19)
        for col,lab in enumerate([['A：留在 CTA 0','B：CTA 1 → CTA 0'],['C：CTA 0 → CTA 1','D：留在 CTA 1']][row]):
            d.rect(180+col*520,199+row*102,490,84,BLUE_LIGHT if row==0 else GOLD_LIGHT)
            d.text(425+col*520,233+row*102,lab,20,anchor='middle')
            d.text(425+col*520,265+row*102,'64 query ×128 KV =16 KiB',18,anchor='middle')
    # Direction labels kept outside matrix, and not mistaken for time bars.
    arrow(d,575,422,325,422,BLUE);d.text(450,454,'B 向左；补齐 CTA 0 的 KV',20,BLUE,anchor='middle')
    arrow(d,685,422,935,422,GOLD);d.text(810,454,'C 向右；补齐 CTA 1 的 KV',20,GOLD,anchor='middle')
    d.text(40,515,'交换后：每个 CTA 的 dS 包含完整256个KV归约元素',23,bold=True)
    for row,(ct,ql,lab) in enumerate([(0,'q0..63','[ A | B ]'),(1,'q64..127','[ C | D ]')]):
        box(d,40,543+row*126,1120,101,f'CTA {ct}：{ql}；dS[64,256] = {lab}',
            ['联合 dQ = dS[128,256] × K[256,128]；本 CTA 得到64×128的输出行'],BLUE_LIGHT if row==0 else GOLD_LIGHT)
    d.text(40,843,'每方向只交换16 KiB：copy 发起者是 compute 的 tidx=0；relay warp 等接收完成。',21,bold=True)
    d.text(40,889,'两边 relay 各向 leader arrive 一次 → leader 等2次到达 → 才发射 dQ UMMA。',21,BLUE)
    d.text(40,935,'K 的 dQ view 也要覆盖全部256 KV，按 feature 分给两 CTA；源码另有 Kt 加载路径。',20)
    d.text(40,981,'dQ 的256个KV贡献在联合 MMA 内完成；其他 KV clusters 的贡献仍需 global reduction。',20)
    d.text(40,1027,'相对两个独立128-KV tiles，各自写完整128行 dQ，配对后每 CTA 只写64行：payload 减半。',20,MUTED)
    return d


def backward():
    d=wide('Backward：5 个 MMA 穿插 P / dS 计算，DSM 交换后再做 dQ',
           'BF16 D128、2-CTA 稳态片段；j是query tile遍历序号；横轴仅表示合法先后和可重叠窗口。',1120)
    t=Timeline(d,x=285,step=175);t.axis(120)
    for y,n in [(176,'warp13 Load'),(275,'warp12 MMA issue'),(367,'Tensor Core 在途'),(475,'WG1+WG2 compute'),(593,'DSM copy 在途'),(696,'warp14 relay'),(797,'WG0 dQ reduce')]:t.lane(y,n)
    t.bar(176,0,2.8,'预取下一个 Q / dO / stats',PANEL)
    for a,b,l in [(0,1,'Sᵀ j+1'),(1,3,'dK j'),(3,4,'dPᵀ j+1'),(4,6,'dQ j'),(6,8,'dV j+1')]:
        t.mark(275,a,l);t.bar(367,a,b,l)
    t.bar(475,1,3.4,'P j+1 → TMEM',BLUE_LIGHT)
    t.bar(475,4,6.2,'计算 dS；等 dQ 后写回',GOLD_LIGHT)
    t.mark(475,6.2,'发起双向 copy',labeldy=73)
    t.bar(593,6.2,7.4,'dS j+1 → peer',GOLD_LIGHT)
    t.mark(696,7.4,'');d.text(1745,720,'双 CTA 到达 → 下轮 dQ 可用',17,BLUE,anchor='end')
    t.bar(797,6,8,'dQ j → shared → global add',GREEN_LIGHT)
    for a,y1,y2 in [(1,405,475),(4,405,475),(6,405,797),(7.4,631,696)]:t.gate(a,y1,y2)
    d.text(40,917,'片段起点前：dS j 已由上一轮生成并启动交换；dQ j 仍须等待本地 dS 与双方 relay。',21,bold=True)
    d.text(40,963,'P j+1 与 dK j 重叠；dS j+1 与 dQ j 重叠；dQ 的搬出 / global add 与后续 MMA 重叠。',22,BLUE)
    d.text(40,1009,'S/P 和 dP/dS 共用 TMEM；D128 的 dQ 还复用 S 区后半段，所以必须严格遵守读完后覆写。',21)
    d.text(40,1055,'Prologue：Sᵀ0 → dPᵀ0 → dV0；Epilogue：最后 dK / dQ，排空 dQ reduction，输出 dK / dV。',20,MUTED)
    return d

FIGURES=[('roles',roles),('operands',operands),('layout',layout),('forward',forward),('exchange',exchange),('backward',backward)]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output_dir',nargs='?',type=Path,default=Path(__file__).parent)
    out=p.parse_args().output_dir;out.mkdir(parents=True,exist_ok=True)
    for name,fn in FIGURES:
        d=fn();f=out/f'fa4_blackwell_{name}.svg';f.write_text(d.svg(),encoding='utf8')
        renderer.W=1800 if 'width="1800"' in d.parts[0] else 1200;renderer.H=d.height
        renderer.render_png(f,f.with_suffix('.png'))
if __name__=='__main__':main()
