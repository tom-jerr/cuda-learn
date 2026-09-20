#!/usr/bin/env python3
"""生成 FA3 跨 Warpgroup Pingpong 与组内两级流水 SVG。"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

W, H = 1600, 900
INK, MUTED, BLUE = "#18212B", "#667085", "#246BCE"
QK, PV, SM, WAIT, FLIGHT = "#F6D58F", "#CFC4E6", "#DCEBFA", "#F2F4F7", "#4B5563"
BG, GRID, BARRIER = "#FBFCFE", "#CBD5E1", "#E5F0FF"


def e(v: object) -> str:
    return html.escape(str(v), quote=True)


class SVG:
    def __init__(self) -> None:
        self.a: list[str] = []

    def add(self, raw: str) -> None:
        self.a.append(raw)

    def text(self, x, y, value, size=18, weight=400, fill=INK,
             anchor="start", italic=False) -> None:
        style = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{style}>{e(value)}</text>')

    def rect(self, x, y, w, h, fill="white", stroke=INK, sw=1.4,
             rx=0, dash=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1.4, dash=None,
             marker=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, stroke=BLUE, sw=1.6, marker="blue-arrow") -> None:
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{m}/>')


def event(s: SVG, x, y, w, label, sub, fill, stroke=INK) -> None:
    s.rect(x, y, w, 58, fill, stroke, 1.4, 5)
    s.text(x + w / 2, y + 24, label, 18, 650, anchor="middle")
    s.text(x + w / 2, y + 46, sub, 13, 400, MUTED, anchor="middle")


def axis(s: SVG, y: int) -> None:
    s.line(220, y, 1510, y, INK, 1.2, marker="arrow")
    s.text(1530, y + 6, "时间", 15, 500, MUTED)


def boundary(s: SVG, x: int, y1: int, y2: int) -> None:
    s.line(x, y1, x, y2, GRID, 1.2, "6 6")


def build() -> str:
    s = SVG()
    s.add(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">FA3 跨 Warpgroup Pingpong 调度与组内两阶段流水线</title>
<desc id="desc">两个 consumer Warpgroup 通过 turn mbarrier 交替发射 WGMMA；每个组内部让 Softmax 与异步 PV WGMMA 重叠。</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{INK}"/></marker>
  <marker id="blue-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{BLUE}"/></marker>
  <style>text {{ font-family: "Droid Sans Fallback", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", Arial, sans-serif; }}</style>
</defs>''')
    s.rect(0, 0, W, H, BG, "none", 0)
    s.text(42, 54, "FA3：跨 Warpgroup Pingpong 调度", 30, 700)
    s.text(42, 84, "示意时序（非等比例）｜1 个 producer WG + 2 个 consumer WG", 16, 400, MUTED)

    # Panel A: cross-WG schedule.
    s.text(42, 132, "A  跨 Warpgroup：turn[2] mbarrier 控制 WGMMA 发射权", 21, 700)
    s.text(52, 194, "Consumer WG 0", 17, 650)
    s.text(52, 314, "Consumer WG 1", 17, 650)
    y0, y1 = 158, 278
    axis(s, 248); axis(s, 368)
    event(s, 220, y0, 150, "QK(n+1)", "WGMMA commit", QK)
    event(s, 370, y1, 150, "QK(n+1)", "WGMMA commit", QK)
    event(s, 520, y0, 150, "PV(n)", "WGMMA commit", PV)
    event(s, 670, y1, 150, "PV(n)", "WGMMA commit", PV)
    event(s, 670, y0, 68, "等待", "<1>", WAIT)
    event(s, 738, y0, 205, "Softmax(n+1)", "CUDA Core / SFU", SM)
    event(s, 943, y0, 68, "等待", "<0>", WAIT)
    event(s, 820, y1, 68, "等待", "<1>", WAIT)
    event(s, 888, y1, 205, "Softmax(n+1)", "CUDA Core / SFU", SM)
    event(s, 1093, y1, 68, "等待", "<0>", WAIT)
    event(s, 1011, y0, 155, "QK(n+2)", "下一轮发射", QK)
    event(s, 1161, y1, 155, "QK(n+2)", "下一轮发射", QK)
    s.rect(520, 216, 491, 27, FLIGHT, FLIGHT, 0)
    s.text(765, 235, "PV(n) 在 Tensor Core 中异步执行", 14, 600, "white", anchor="middle")
    s.rect(670, 336, 491, 27, FLIGHT, FLIGHT, 0)
    s.text(915, 355, "PV(n) 在 Tensor Core 中异步执行", 14, 600, "white", anchor="middle")
    for x in (370, 520, 670, 820, 1011, 1161):
        boundary(s, x, 145, 378)
    s.text(610, 145, "token：WG0.QK → WG1.QK → WG0.PV → WG1.PV → …", 16, 650, BLUE, anchor="middle")
    s.path("M 1310 195 L 1215 210 L 1030 210", BLUE)
    s.text(1325, 177, "WG1 做 Softmax 时", 14, 650, BLUE)
    s.text(1325, 198, "WG0 已可进入下一轮", 14, 650, BLUE)

    # Panel B: intra-consumer two-level pipeline.
    s.line(42, 410, 1558, 410, GRID, 1)
    s.text(42, 458, "B  单个 Consumer 内：older QK + younger PV 的两级异步流水", 21, 700)
    s.text(52, 526, "Consumer WG k", 17, 650)
    y = 490
    axis(s, 594)
    event(s, 220, y, 190, "QK(next)", "older WGMMA group", QK)
    event(s, 410, y, 190, "PV(cur)", "younger WGMMA group", PV)
    event(s, 600, y, 110, "等待", "wait_group<1>", WAIT)
    event(s, 710, y, 270, "Softmax(next)", "scores 可见 · exp2", SM)
    event(s, 980, y, 110, "等待", "wait_group<0>", WAIT)
    event(s, 1090, y, 180, "stage_empty", "mbarrier arrive", BARRIER, BLUE)
    event(s, 1270, y, 180, "缩放 O", "× alpha(next)", "white")
    s.rect(410, 548, 680, 34, FLIGHT, FLIGHT, 0)
    s.text(750, 571, "PV(cur) 仍可在 Tensor Core 中飞行", 15, 650, "white", anchor="middle")
    boundary(s, 600, 475, 605); boundary(s, 1090, 475, 605)
    s.line(615, 470, 1075, 470, BLUE, 1.5)
    s.path("M 615 470 L 627 464 M 615 470 L 627 476", BLUE, 1.5, None)
    s.path("M 1075 470 L 1063 464 M 1075 470 L 1063 476", BLUE, 1.5, None)
    s.text(845, 461, "Softmax 与 younger PV 的异步执行区间重叠", 15, 650, BLUE, anchor="middle")

    # Dependency summary.
    s.text(42, 660, "源码对应关系", 20, 700)
    rows = [
        ("发射令牌", "pingpong_before / pingpong_after", "turn[2]：只串行化两个 consumer 的 WGMMA commit 顺序"),
        ("分数就绪", "wgmma_wait_group<1>", "等待 older QK；允许 younger PV 继续 in-flight"),
        ("完全完成", "wgmma_wait_group<0>", "Softmax 后等待 PV 完成，再释放当前 KV stage"),
        ("双缓冲复用", "stage_empty[2]", "两个 consumer 都 arrive 后，producer 才能覆盖 K/V stage"),
    ]
    for i, (name, code, desc) in enumerate(rows):
        yy = 692 + i * 43
        s.rect(42, yy - 25, 1516, 36, "white" if i % 2 == 0 else "#F7F9FC", GRID, 0.8, 3)
        s.text(62, yy, name, 15, 650)
        s.text(250, yy, code, 15, 600, BLUE)
        s.text(610, yy, desc, 15, 400)

    s.add("</svg>")
    return "\n".join(s.a) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output", nargs="?", type=Path,
                   default=Path(__file__).with_suffix(".svg"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(), encoding="utf-8")


if __name__ == "__main__":
    main()
