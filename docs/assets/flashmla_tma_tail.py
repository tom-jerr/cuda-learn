#!/usr/bin/env python3
"""Dense FlashMLA TMA tail clearing, derived from c2067be fill_oob_V.

Static token/feature layout; arrows label traversal, not measured time.
Run with an optional output directory. Uses the existing SVG/PNG helpers.
"""
import argparse
from pathlib import Path
from cute_tile_views import Drawing, BLUE, BLUE_LIGHT, GOLD_LIGHT, MUTED
from flashmla_initial_figures import arrow, box, RED, RED_LIGHT
import cute_r2s_retile as renderer


def figure():
    d = Drawing('新版 TMA：整页加载后，按有效长度清零 shared V',
                '固定 c2067be 的 BF16 dense decode；L=130，尾页起点128，valid=2。行高为示意。', 1120)
    d.text(40, 123, '一页 K：64 tokens × 576 features；9 次 TMA，每次 64×64 BF16 = 8192 B。', 22, bold=True)
    x0, y0, cw = 140, 197, 108
    for j in range(9):
        d.text(x0+j*cw+cw/2, 170, f'j={j}', 18, anchor='middle')
        for i, (label, h, fill) in enumerate([('有效', 56, BLUE_LIGHT), ('有效', 56, BLUE_LIGHT), ('NaN?', 112, RED_LIGHT)]):
            y = y0 + (0 if i == 0 else 56 if i == 1 else 112)
            d.rect(x0+j*cw, y, cw, h, fill)
            d.text(x0+j*cw+cw/2, y+h/2+7, label, 19, anchor='middle')
    for y, s in [(232, 'r=0'), (288, 'r=1'), (365, 'r=2..63')]:
        d.text(125, y, s, 18, anchor='end')
    d.text(140, 449, 'f=0..255：WG0 清零', 20, BLUE)
    d.text(572, 449, 'f=256..511：WG1 清零', 20, BLUE)
    d.text(1058, 449, 'RoPE', 20, anchor='middle')
    arrow(d, 140, 473, 1003, 473)
    d.text(140, 508, '只清红色区域的前8条带；RoPE 不参与 PV，不需要为 PV 清零。', 21)
    d.text(40, 553, '物理页64行都在 descriptor 范围内；TMA 不认识每个请求的 L=130。', 22, RED, True)
    box(d, 40, 577, 1120, 141, 'fill_oob_V：每个 WG 把自己的256维 V 重解释成 int64 包', [
        '256 BF16 / 4 = 64 个包；128线程分两组，每轮覆盖两条token行。',
        'u=0..127：p=u%64；r=valid+u/64+2a；a 向下递增，直到 r≥64。',
        '每次写0覆盖4个相邻BF16；在这个例子中，每线程执行31次8-byte写。'
    ], GOLD_LIGHT)
    d.table(40, 742, [265, 285, 285, 285], [
        ['WG 内线程 u', 'feature 包 p', '首轮清零 token r', '下一轮：向下 +2'],
        ['0..63', '0..63，各4 BF16', '2', '4, 6, …, 62'],
        ['64..127', '0..63，各4 BF16', '3', '5, 7, …, 63'],
    ], rh=44)
    d.text(40, 914, 'int64 底层 layout：((16,4),(8,8)):((1,1024),(16,128))。', 21, bold=True)
    d.text(40, 953, '完整包地址：E(r,p)=1024⌊p/16⌋+16r+((p mod16) XOR 2(r mod8))。', 20)
    d.text(40, 990, '相对本WG半块基址，单位为int64；对应BF16地址 W(r,4p)=4E(r,p)。', 20)
    d.text(40, 1042, 'TMA完成且相关QK读完 → 写零 → proxy fence / WG协作 → PV读取', 22, BLUE, True)
    d.text(40, 1084, 'S 中覆盖 mask 解决概率；V 中覆盖0解决 0×NaN。有效输入中的NaN不在清理范围内。', 18, MUTED)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir', nargs='?', type=Path, default=Path(__file__).parent)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    d = figure()
    target = args.output_dir / 'flashmla_tma_tail.svg'
    target.write_text(d.svg(), encoding='utf-8')
    renderer.W, renderer.H = 1200, d.height
    renderer.render_png(target, target.with_suffix('.png'))
    print(target)


if __name__ == '__main__':
    main()
