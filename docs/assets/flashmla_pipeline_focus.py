#!/usr/bin/env python3
"""Source-derived FlashMLA 15f13e5 layouts and legal schematic schedules.
All times are ordinal drawing positions, not measurements or latency predictions.
Default writes beside this file; optional positional output directory.
"""
import argparse
from pathlib import Path
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD, GOLD_LIGHT, PANEL, MUTED
from flashmla_initial_figures import arrow, box
import cute_r2s_retile as renderer

GREEN, GREEN_LIGHT = '#256747', '#E1F1E8'


def wide(title, subtitle, height):
    d=Drawing(title,subtitle,height)
    d.parts=[s.replace('width="1200"','width="1800"').replace(f'viewBox="0 0 1200 {height}"',f'viewBox="0 0 1800 {height}"') for s in d.parts]
    return d


def axes():
    d=Drawing('三种“切半”：输出列拼接、序列求和、head拼接',
              '统一使用同一max基准下的P；Õ表示尚未除以行分母的输出。',1000)
    box(d,40,115,1120,102,'① 输出列切半：P 完全相同；没有切开 token 归约',
        ['P[64,N] × [ V_L[N,256] | V_R[N,256] ] = [ Õ_L[64,256] | Õ_R[64,256] ]'],BLUE_LIGHT)
    d.text(40,263,'把序列再切成两个 token blocks，可以看成一个2×2的乘积网格：',22,bold=True)
    d.table(40,294,[270,425,425],[
        ['同一个输出的贡献','输出 features 0..255','输出 features 256..511'],
        ['token block 0','P0 × V0L','P0 × V0R'],
        ['token block 1','P1 × V1L','P1 × V1R'],
    ],rh=67)
    arrow(d,481,506,481,554)
    arrow(d,906,506,906,554)
    d.text(481,588,'沿 token 方向相加 → Õ_L',21,BLUE,True,anchor='middle')
    d.text(906,588,'沿 token 方向相加 → Õ_R',21,GOLD,True,anchor='middle')
    d.rect(310,625,425,70,BLUE_LIGHT);d.rect(735,625,425,70,GOLD_LIGHT)
    d.text(523,668,'O_L = Õ_L / l',23,anchor='middle')
    d.text(948,668,'O_R = Õ_R / l',23,anchor='middle')
    d.text(40,735,'列方向拼接 → O[64,512]；两边的 l 必须相同。',23,bold=True)
    box(d,40,766,1120,94,'② 序列 split：局部归一化输出不能直接相加',
        ['每段得到 O_s、LSE_s；按 exp(LSE_s − LSE_total) 加权相加。'],GOLD_LIGHT)
    box(d,40,883,1120,84,'③ Cluster 的 head 切半：CTA0 heads0..63；CTA1 heads64..127',
        ['每个head各有自己的softmax；最终沿head维拼接，不跨CTA做softmax归约。'],GREEN_LIGHT)
    return d


def dense_layout():
    d=Drawing('Dense TMA：global行主序 → shared条带 + SW128 → MMA view',
              '固定一页、一个KV head；global示例无行padding；BF16 element单位，字节数另标。',1120)
    d.table(40,115,[220,430,470],[
        ['对象','shape','stride / 物理单位'],
        ['global K页','(64,576)','(576,1) BF16；每行1152B'],
        ['一次 TMA tile','(64,64)','源：每行取64值，跨到下一行仍+576'],
        ['shared K','((8,8),(64,9))','((64,512),(1,4096)) + SW128'],
    ],rh=47)
    d.text(40,348,'global：一次 j=0 传输，取每行前64列；不是连续8192B的 memcpy。',21,bold=True)
    for r in range(4):
        for j in range(9):
            d.rect(100+j*112,374+r*40,112,40,BLUE_LIGHT if j==0 else PANEL)
            d.text(156+j*112,400+r*40,str(r*576+j*64),16,anchor='middle')
        d.text(87,400+r*40,f'r{r}',17,anchor='end')
    d.text(40,565,'源右移64列 +64；源下移1行 +576。只画前4行，数字为每格源起点。',19,MUTED)
    arrow(d,600,583,600,629)
    d.text(40,663,'shared：每个64×64条带单独占4096值；TMA把包写入swizzled位置。',21,bold=True)
    for j in range(9):
        d.rect(100+j*112,690,112,72,GOLD_LIGHT if j==8 else BLUE_LIGHT)
        d.text(156+j*112,719,f'j{j}',19,anchor='middle')
        d.text(156+j*112,747,str(j*4096),16,anchor='middle')
    d.text(40,798,'目的右移64列 +4096；目的下移8行 +512；包内8个BF16顺序不变。',20,BLUE)
    d.table(40,831,[260,425,435],[
        ['同一元素 (r,f)','global offset','shared 实际 offset（已应用XOR）'],
        ['(0,64)','64','4096'],
        ['(1,0)','576','72 = 64+8'],
        ['(1,8)','584','64 = 64+0'],
    ],rh=43)
    d.text(40,1050,'SS WGMMA：直接用shared descriptor；V(f,r)=K(r,f)，只换view，不搬运。',20,bold=True)
    d.text(40,1091,'r=1时逻辑16B包0、1交换；这属于SW128地址置换，不是矩阵转置。',19,MUTED)
    return d


class Timeline:
    def __init__(self,d,x=230,step=93): self.d=d;self.x=x;self.step=step
    def xx(self,t):return self.x+t*self.step
    def lane(self,y,name):
        self.d.text(40,y+27,name,20,bold=True)
        self.d.parts.append(f'<path d="M{self.x},{y+48} H1755" stroke="#CBD3DE" fill="none"/>')
    def bar(self,y,a,b,label,fill=PANEL,color=None,h=38):
        assert b>a
        self.d.rect(self.xx(a),y,self.step*(b-a),h,fill)
        self.d.text(self.xx((a+b)/2),y+h/2+6,label,17,color or '#172033',anchor='middle')
    def mark(self,y,t,label,color=BLUE,labeldy=17):
        x=self.xx(t)
        self.d.parts.append(f'<path d="M{x},{y-4} V{y+42}" stroke="{color}" stroke-width="2"/>')
        if label:self.d.text(x+5,y+labeldy,label,16,color)
    def gate(self,t,y1,y2,label=''):
        x=self.xx(t)
        self.d.parts.insert(4,f'<path d="M{x},{y1} V{y2}" stroke="{BLUE}" stroke-dasharray="4 4" fill="none"/>')
        if label:self.d.text(x+5,y2-5,label,16,BLUE)
    def axis(self,y):
        arrow(self.d,self.x,y,1755,y)
        self.d.text(1755,y+31,'先后关系 →；非测量，不按比例',18,MUTED,anchor='end')


def dense_pipeline():
    d=wide('Dense：两 WG 的发射、Tensor Core 未完成窗口与 TMA 重叠',
           '一个满足源码依赖的代表性时序；K0/K1是槽位，block0..3是数据。灰条表示可能未完成，不代表独占计算单元。',1180)
    t=Timeline(d)
    t.axis(118)
    lanes=[(180,'WG0 标量/发射'),(265,'WG0 PV 在途'),(350,'WG0 QK 在途'),(455,'WG1 标量/发射'),(540,'WG1 PV 在途'),(625,'WG1 QK 在途'),(740,'TMA → 左半'),(825,'TMA → 右半')]
    for y,n in lanes:t.lane(y,n)
    t.bar(180,0,1,'SM0',BLUE_LIGHT);t.mark(180,1,'PV0L↑')
    t.bar(180,3,4,'P0×a1→sP0',BLUE_LIGHT);t.mark(180,4.5,'PV1L↑')
    t.bar(180,6,7.6,'QK2 j0..3↑',BLUE_LIGHT);t.mark(180,8,'wait<4>')
    t.bar(180,9.5,11,'QK2 j4..8↑',BLUE_LIGHT);t.bar(180,12,13,'SM2',BLUE_LIGHT)
    t.bar(265,1,3,'PV0L 在途');t.bar(265,4.5,8,'PV1L 在途')
    t.bar(350,6,12,'QK2：左4组先发，后5组接续；最终 wait<0>')
    t.bar(455,2,3,'SM1',GOLD_LIGHT);t.mark(455,3.6,'PV1R↑')
    t.mark(455,4.4,'PV0R↑')
    t.mark(455,5.6,'wait<1>',labeldy=-12);t.mark(455,7,'wait<0>',labeldy=-12)
    t.bar(455,7.7,9.3,'QK3 j4..8↑',GOLD_LIGHT);t.bar(455,11,12.5,'QK3 j0..3↑',GOLD_LIGHT)
    t.bar(455,14,15,'SM3',GOLD_LIGHT)
    t.bar(540,3.6,5.6,'PV1R 在途');t.bar(540,5.6,7,'PV0R 尾部')
    # PV0R begins before PV1R completes; a separate narrow sub-band records its full lifetime.
    t.bar(584,4.4,7,'PV0R完整窗口',PANEL,h=24)
    t.bar(625,0,2,'QK1→完成');t.bar(625,7.7,11,'QK3 右5组');t.bar(625,11,14,'左4组→wait<0>')
    t.bar(740,3,6,'block2 j0..3 → K0',GREEN_LIGHT)
    t.bar(740,8,11,'block3 j0..3 → K1',GREEN_LIGHT)
    t.bar(825,5.6,7.5,'block3 j4..8',GREEN_LIGHT)
    t.bar(825,7.5,9.5,'block2 j4..8',GREEN_LIGHT)
    # The second transfer is issued at 7; the band visualizes an allowed later service window.
    t.mark(825,7,'block2 TMA↑',labeldy=-12)
    t.gate(3,306,740);t.gate(8,306,740)
    t.gate(5.6,578,825);t.gate(7,608,825)
    d.text(40,933,'SM = softmax；↑ = 发射。WG1的SM1还要等WG0发布scale0；SM2/SM3也保持同样依赖。',21)
    d.text(40,978,'P0在SM1更新max后乘a1，再交给WG1；WG0等到WG1发射PV0R后，才发射PV1L。',21)
    d.text(40,1023,'wait<4>只保证旧PV1L结束；4个较新QK提交组可仍在途，所以TMA能与QK重叠。',21,BLUE)
    d.text(40,1068,'图内对不同传输/计算的相对长短只选取一种合法情况；实际完成顺序与重叠长度由硬件运行决定。',20,MUTED)
    d.text(40,1113,'首块：WG0先等完整K0后做SS QK；本图从该QK已完成、WG1的QK1仍可在途的位置展开。',20,MUTED)
    d.text(40,1154,'右半TMA含RoPE；到达后仍逐条带等待barrier。本图为了可读性将同半边条带的服务窗口合并。',20,MUTED)
    return d


def sparse_layout():
    d=Drawing('Hopper sparse FP8：gather → 寄存器反量化 → 两份 INTER shared',
              'V32 / cluster=2。这里KV路径没有TMA；TMA用于Q加载及相关输出搬运。',1110)
    box(d,40,112,1120,117,'HBM：indices先选物理token，再从656B记录取数据',
        ['[512B E4M3 latent | 16B FP32 scales | 128B BF16 RoPE]', 'producer每lane取16个FP8；4个lane共同覆盖一个token的64维。'],BLUE_LIGHT)
    arrow(d,600,235,600,278)
    box(d,40,292,540,122,'CTA0 producer：selected rows0..31',
        ['FP8 + scale → 16 BF16 / lane / 轮', '拆成两个8-BF16包，各16B'],BLUE_LIGHT)
    box(d,620,292,540,122,'CTA1 producer：selected rows32..63',
        ['与CTA0相同的feature分工', '只处理另一半tokens'],GOLD_LIGHT)
    d.text(40,462,'INTER平铺顺序：先向下填满64×8，再向右到下一个8-feature条带。',21,bold=True)
    d.table(40,485,[240,290,290,300],[
        ['例：selected row5','features0..7','features8..15','下一64-feature块'],
        ['shared位置','40..47','552..559','起点 40+4096'],
        ['来自同一次16B FP8读','解出前8 BF16','解出后8 BF16','下一次16B FP8读'],
    ],rh=48)
    d.text(40,668,'同一个BF16包：普通store写本CTA；st.async写peer CTA的相同offset。',21,BLUE)
    box(d,40,704,540,131,'CTA0 shared K：64×576 BF16',
        ['rows0..31：local producer写入', 'rows32..63：CTA1经DSM写入', 'V左/右半只是K的前512维view'],BLUE_LIGHT)
    box(d,620,704,540,131,'CTA1 shared K：64×576 BF16',
        ['rows0..31：CTA0经DSM写入', 'rows32..63：local producer写入', '和CTA0内容相同，Q heads不同'],GOLD_LIGHT)
    d.text(40,894,'每方向36KiB：32 tokens ×576 BF16 ×2 bytes；两个CTA都要等待完整64行。',21)
    d.text(40,940,'INTER element offset = 8r + 512⌊f/8⌋ + (f mod8)',22,mono=True)
    d.text(40,986,'向右256 features：+16384 BF16；sV_R与sV_L共享同一K分配。',21)
    d.text(40,1032,'反量化与重新排布由线程完成；DSM保持这些BF16包的目标offset，不再做反量化。',20,MUTED)
    d.text(40,1075,'计算块的selected row与原序列/物理页的row不同：只有查indices后才知道源地址。',20,MUTED)
    return d


def sparse_pipeline():
    d=wide('Hopper sparse cluster：producer重叠下一块；四个consumer共同释放旧槽',
           '一种合法示意时序，不按比例。C0/C1是CTA；B0/B1/B2是64-token计算块；消费者窄条为WGMMA未完成窗口。',1240)
    t=Timeline(d,x=280,step=89)
    t.axis(118)
    for y,n in [(180,'C0 producer WG2'),(320,'C1 producer WG2'),(480,'C0 consumer WG0'),(640,'C0 consumer WG1'),(800,'C1 consumer WG0'),(960,'C1 consumer WG1')]:t.lane(y,n)
    for y in [180,320]:
        t.bar(y,0,2,'gather+dequant B0',BLUE_LIGHT)
        t.bar(y+48,.6,2.5,'DSM→peer B0',PANEL,h=27)
        t.bar(y,2,4,'gather+dequant B1',GOLD_LIGHT)
        t.bar(y+48,2.6,4.8,'DSM→peer B1',PANEL,h=27)
        t.bar(y,4,9,'等 slot0 avail：必须收到4次释放',PANEL)
        t.bar(y,9,11,'gather+dequant B2',GREEN_LIGHT)
        t.bar(y+48,9.6,11.8,'DSM→peer B2',PANEL,h=27)
    # Consumers from two CTAs can have different progress. Upper bars are scalar/issue events;
    # lower bars are asynchronous MMA windows, thus no claimed identical hardware duration.
    for y,delay in [(480,0),(800,.4)]:
        t.mark(y,3+delay,'QK0↑')
        t.bar(y+48,3+delay,5+delay,'QK0 在途',PANEL,h=27)
        t.bar(y,5+delay,6+delay,'SM0/P0',BLUE_LIGHT)
        t.mark(y,6+delay,'PV0L↑')
        t.bar(y+48,6+delay,7.5+delay,'PV0L',PANEL,h=27)
        t.mark(y,7.5+delay,'release0 / QK1↑')
        t.bar(y+48,7.5+delay,9.5+delay,'QK1',PANEL,h=27)
        t.bar(y,10+delay,11+delay,'SM1/P1',GOLD_LIGHT)
        t.mark(y,11+delay,'PV1L↑')
        t.bar(y+48,11+delay,13+delay,'PV1L',PANEL,h=27)
    for y,start,end in [(640,6,8.5),(960,6.4,9)]:
        t.bar(y,0,start,'等本CTA P0 + scale0',PANEL)
        t.mark(y,start,'scale O / PV0R↑')
        t.bar(y+48,start,end,'PV0R 在途',PANEL,h=27)
        t.mark(y,end,'release0 / P-free')
        t.bar(y,11.5,12.8,'取P1/scale',GOLD_LIGHT)
        t.bar(y+48,12.8,15,'PV1R 在途',PANEL,h=27)
    t.gate(9,215,1065)
    d.text(t.xx(9)+8,1103,'slot0：4个consumer都完成PV，B2才可覆盖',20,BLUE)
    d.text(40,1150,'K-ready = local_ready(128 arrivals) + remote_ready(36KiB)；K-available = C0.WG0/1 + C1.WG0/1。',20)
    d.text(40,1195,'P-free仅在本CTA内握手：WG0可先算下一块QK；写下一份shared P/scale前必须等WG1释放。',20,MUTED)
    return d


def blackwell_layout():
    d=Drawing('Blackwell head64：TMA raw layout 与反量化后的 MMA layout',
              'V32 NoPE512；这里是单个head64路径，不使用Hopper双CTA的DSM cross-over。',1120)
    box(d,40,113,1120,102,'HBM tensor map：一行是一条物理token的NoPE部分',
        ['dtype=INT64（8个FP8字节一包）；shape=(64, physical_rows)；行byte stride=656。'],BLUE_LIGHT)
    d.text(40,258,'gather4 输入四个独立row坐标：例 [17,400,-1,23]；无效行触发零填充。',21,bold=True)
    arrow(d,600,280,600,325)
    box(d,40,338,1120,117,'raw shared：64个选中token紧密排列，每行512个FP8',
        ['shape=(64,512), stride=(512,1)，单位byte；无SW128 XOR。', 'global行stride656 → shared行stride512；64行共32KiB；16次gather4。'],GOLD_LIGHT)
    arrow(d,600,463,600,510)
    box(d,40,524,1120,146,'dequant WG：每8线程处理一个token的64-feature块',
        ['每线程读8个FP8 → 乘scale → 写8个BF16。', '128线程 =16组；每组做4个token，覆盖64行；沿feature做8轮。', '写入64×512 BF16 SW128：8个64-feature条带，共64KiB。'],GREEN_LIGHT)
    d.table(40,710,[245,395,480],[
        ['同一个逻辑移动','raw FP8 shared（byte）','BF16 SW128 shared（element）'],
        ['向下16个token','+16×512=8192','+16×64=1024'],
        ['向右64 features','+64','+64×64=4096'],
        ['向右8 features','+8','包位置由行号XOR决定'],
    ],rh=46)
    d.text(40,945,'MMA view再变化：Q进TMEM；K以dual-GEMM view解释；PV以转置V view解释。',20,bold=True)
    d.text(40,991,'RoPE由独立gather路径直接写BF16 SW64；V32可先算RoPE QK，等待NoPE反量化。',20)
    d.text(40,1037,'TMA只搬字节并按descriptor排布；FP8→BF16与乘scale仍由线程执行。',21,BLUE)
    d.text(40,1082,'MODEL1的payload/scale组织和RoPE布局不同；本图地址只对应V32。',19,MUTED)
    return d


def blackwell_pipeline():
    d=wide('Blackwell head64：索引、双TMA流、反量化、TC与softmax',
           '满足源码依赖的一种示意时序；非测量、不按比例。TC灰条是异步在途窗口；R=RoPE，N=NoPE。',1180)
    t=Timeline(d,x=235,step=94)
    t.axis(115)
    for y,n in [(180,'warp7 索引/scale'),(275,'warp5 TMA NoPE'),(380,'warp6 TMA RoPE'),(485,'WG2 反量化'),(605,'warp4 MMA发射'),(710,'TC 在途'),(825,'WG0 softmax')]:t.lane(y,n)
    t.bar(180,0,1,'idx0',BLUE_LIGHT);t.bar(180,1,2,'idx1',GOLD_LIGHT);t.bar(180,2,3,'idx2',GREEN_LIGHT)
    t.bar(275,1,3,'gather N0 →raw0',BLUE_LIGHT);t.bar(275,3,5,'gather N1 →raw1',GOLD_LIGHT)
    t.bar(275,6,8,'gather N2 →raw0',GREEN_LIGHT)
    t.bar(380,1,2,'gather R0',BLUE_LIGHT);t.bar(380,2,3,'gather R1',GOLD_LIGHT)
    t.bar(380,7,8.5,'gather R2',GREEN_LIGHT)
    t.bar(485,3,6,'dequant N0 →BF16 slot0',BLUE_LIGHT)
    t.bar(485,6,9,'dequant N1 →BF16 slot1',GOLD_LIGHT)
    t.bar(485,10.5,13.5,'dequant N2 →BF16 slot0',GREEN_LIGHT)
    t.mark(605,2,'QK0-R↑');t.mark(605,6,'QK0-N↑');t.mark(605,8.5,'PV0↑')
    t.mark(605,10.5,'QK1-R↑');t.mark(605,11.5,'QK1-N↑');t.mark(605,14,'PV1↑')
    t.bar(710,2,3,'QK0-R');t.bar(710,6,7,'QK0-N');t.bar(710,8.5,10.5,'PV0')
    t.bar(710,10.5,12.5,'QK1 R+N');t.bar(710,14,15.8,'PV1')
    t.bar(825,7,8.5,'合并/SM0/P0',BLUE_LIGHT)
    t.bar(825,12.5,14,'合并/SM1/P1',GOLD_LIGHT)
    t.gate(6,523,275);t.gate(7,748,380);t.gate(10.5,748,485)
    d.text(40,945,'三种释放点不同：raw0在反量化完成后释放；R0在QK0完成后释放；BF16 N0要等PV0完成。',22,bold=True)
    d.text(40,993,'所以N2可以先被TMA装入raw0，仍须等PV0结束才能反量化并覆盖BF16 slot0。',21,BLUE)
    d.text(40,1041,'索引有4槽、raw/dequant KV有2槽；本图只展开3个索引块。Q的TMA→TMEM prologue未画。',20,MUTED)
    d.text(40,1089,'MMA warp按源码顺序等待P-ready后发射PV，再进入下一块QK；不把QK1提前画到PV0之前。',20,MUTED)
    d.text(40,1137,'V32的RoPE不属于V；MODEL1的RoPE也参与PV，因而其覆写门槛必须改为PV完成。',20,MUTED)
    return d


FIGURES=[('split_axes',axes),('dense_tma_flow',dense_layout),('dense_pipeline',dense_pipeline),
         ('sparse_layout_flow',sparse_layout),('cluster_pipeline',sparse_pipeline),
         ('blackwell_layout_flow',blackwell_layout),('blackwell_pipeline',blackwell_pipeline)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir',nargs='?',type=Path,default=Path(__file__).parent)
    out=p.parse_args().output_dir;out.mkdir(parents=True,exist_ok=True)
    for name,fn in FIGURES:
        d=fn();f=out/f'flashmla_focus_{name}.svg';f.write_text(d.svg(),encoding='utf8')
        renderer.W=1800 if 'width="1800"' in d.parts[0] else 1200
        renderer.H=d.height
        renderer.render_png(f,f.with_suffix('.png'))

if __name__=='__main__':main()
