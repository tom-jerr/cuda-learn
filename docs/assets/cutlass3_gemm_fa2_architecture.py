#!/usr/bin/env python3
"""Generate deterministic, editable CUTLASS 3.x GEMM/FA2 SVG diagrams."""

from __future__ import annotations

import argparse
import html
from pathlib import Path


INK = "#172033"
MUTED = "#5D687A"
GRID = "#CBD3DE"
PAPER = "#F7F9FC"
WHITE = "#FFFFFF"
BLUE = "#174EA6"
BLUE_LIGHT = "#DCE8FF"
GOLD = "#995600"
GOLD_LIGHT = "#FCE9C8"
GREEN = "#21654D"
GREEN_LIGHT = "#DEF1E8"
PURPLE = "#7142A0"
PURPLE_LIGHT = "#EEE4F7"
RED = "#A33A3A"
RED_LIGHT = "#F8E0E0"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class Svg:
    def __init__(self, width: int, height: int, title: str, description: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{esc(title)}</title>",
            f"<desc id=\"desc\">{esc(description)}</desc>",
            "<defs>",
            f'<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>',
            f'<marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>',
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x: int, y: int, w: int, h: int, fill: str = WHITE,
             stroke: str = GRID, radius: int = 14, stroke_width: int = 2,
             dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr}/>'
        )

    def line(self, x1: int, y1: int, x2: int, y2: int, color: str = INK,
             width: int = 2, arrow: bool = False, dash: str | None = None) -> None:
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        if arrow and color == BLUE:
            marker = ' marker-end="url(#arrow-blue)"'
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"{marker}{dash_attr}/>'
        )

    def path(self, d: str, color: str = INK, width: int = 2,
             arrow: bool = False, dash: str | None = None) -> None:
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        if arrow and color == BLUE:
            marker = ' marker-end="url(#arrow-blue)"'
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{marker}{dash_attr}/>'
        )

    def text(self, x: int, y: int, value: str, size: int = 22,
             color: str = INK, weight: int = 500, anchor: str = "start",
             family: str = "DejaVu Sans") -> None:
        self.add(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-family="{family}" font-weight="{weight}" text-anchor="{anchor}">'
            f'{esc(value)}</text>'
        )

    def multiline(self, x: int, y: int, lines: list[str], size: int = 20,
                  color: str = INK, weight: int = 500, gap: int = 28,
                  anchor: str = "start") -> None:
        self.add(f'<text x="{x}" y="{y}" fill="{color}" font-family="DejaVu Sans" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">')
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else gap
            self.add(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
        self.add("</text>")

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>", ""])


def label_box(svg: Svg, x: int, y: int, w: int, h: int, title: str,
              detail: str, fill: str, stroke: str) -> None:
    svg.rect(x, y, w, h, fill, stroke, 12, 2)
    svg.text(x + 18, y + 30, title, 20, stroke, 700)
    svg.text(x + 18, y + 58, detail, 16, MUTED, 500)


def architecture_svg() -> str:
    s = Svg(
        1600, 980,
        "CUTLASS 3.x execution architecture for Ampere GEMM and FA2",
        "Static composition diagram showing host adapter, scheduler, mainloop pipeline, epilogue, and the custom CuTe FA2 composition.",
    )
    s.text(70, 62, "CUTLASS 3.x on Ampere · Execution architecture", 34, INK, 750)
    s.text(70, 96, "Shared CuTe atoms: a 3.x GEMM collective and a custom FA2 attention loop", 19, MUTED, 500)

    # GEMM panel.
    s.rect(50, 130, 1500, 405, WHITE, GRID, 20, 2)
    s.text(78, 171, "A. BF16 GEMM · CUTLASS 3.x kernel composition", 25, BLUE, 750)
    label_box(s, 82, 204, 230, 82, "Host adapter", "GemmUniversalAdapter", BLUE_LIGHT, BLUE)
    label_box(s, 360, 204, 246, 82, "Kernel shell", "GemmUniversal<M,N,K,L>", BLUE_LIGHT, BLUE)
    s.line(315, 245, 350, 245, BLUE, 3, True)

    s.rect(650, 187, 850, 135, PAPER, BLUE, 16, 2)
    s.text(676, 219, "Kernel = scheduler + mainloop + epilogue", 20, INK, 700)

    label_box(s, 680, 237, 230, 70, "CTA scheduler", "grid(m,n,batch) · direct", PURPLE_LIGHT, PURPLE)
    label_box(s, 935, 237, 300, 70, "CollectiveMainloop", "Sm80CpAsync<3>", GOLD_LIGHT, GOLD)
    label_box(s, 1260, 237, 210, 70, "CollectiveEpilogue", "FP32 → BF16", GREEN_LIGHT, GREEN)
    s.path("M606 245 H630 V272 H670", BLUE, 3, True)
    s.line(910, 272, 925, 272, INK, 2, True)
    s.line(1235, 272, 1250, 272, INK, 2, True)

    # Mainloop internal strip.
    s.text(82, 358, "Inside CollectiveMainloop · 128×128×32 · 4 warps", 18, GOLD, 700)
    label_box(s, 82, 377, 236, 82, "Gmem TiledCopy", "128-bit cp.async", GOLD_LIGHT, GOLD)
    label_box(s, 355, 377, 236, 82, "SMEM ring", "A/B × 3 stages", GOLD_LIGHT, GOLD)
    label_box(s, 628, 377, 236, 82, "SmemCopyAtom", "ldmatrix N / T", GOLD_LIGHT, GOLD)
    label_box(s, 901, 377, 180, 82, "TiledMMA", "m16n8k16", BLUE_LIGHT, BLUE)
    label_box(s, 1118, 377, 352, 82, "Accumulator + epilogue", "FP32 → LinearCombination → BF16", GREEN_LIGHT, GREEN)
    s.line(318, 418, 345, 418, GOLD, 2, True)
    s.line(591, 418, 618, 418, GOLD, 2, True)
    s.line(864, 418, 891, 418, GOLD, 2, True)
    s.line(1081, 418, 1108, 418, INK, 2, True)
    s.path("M1085 307 V349 H745 V367", GOLD, 2, True, "6 5")
    s.text(82, 492, "Prologue", 17, GOLD, 700)
    s.text(178, 492, "fill stage 0/1", 17, MUTED, 500)
    s.text(350, 492, "Mainloop", 17, GOLD, 700)
    s.text(448, 492, "copy(next) ∥ ldmatrix + mma(current)", 17, MUTED, 500)

    # FA2 panel.
    s.rect(50, 565, 1500, 345, WHITE, GRID, 20, 2)
    s.text(78, 606, "B. FlashAttention-2 · custom CUTLASS 3.x CuTe kernel", 25, PURPLE, 750)
    s.text(78, 636, "No turnkey SM80 FA2 collective: compose the algorithm from TiledCopy and TiledMMA atoms", 17, MUTED, 500)

    label_box(s, 82, 681, 198, 96, "Scheduler", "CTA = (Q tile,H,B)", PURPLE_LIGHT, PURPLE)
    label_box(s, 320, 681, 198, 96, "Prologue", "Q + K₀/V₀ → SMEM", GOLD_LIGHT, GOLD)
    label_box(s, 558, 681, 198, 96, "QK mainloop", "FP16 MMA → FP32 S", BLUE_LIGHT, BLUE)
    label_box(s, 796, 681, 198, 96, "Online softmax", "row max/sum + mask", RED_LIGHT, RED)
    label_box(s, 1034, 681, 198, 96, "PV mainloop", "P(FP16) × V(FP16)", BLUE_LIGHT, BLUE)
    label_box(s, 1272, 681, 198, 96, "Epilogue", "O/l → FP16 store", GREEN_LIGHT, GREEN)
    for x in (280, 518, 756, 994, 1232):
        s.line(x + 6, 729, x + 30, 729, INK, 2, True)

    s.rect(320, 813, 912, 62, GOLD_LIGHT, GOLD, 12, 2)
    s.text(342, 839, "Pipeline · 2-stage K/V ping-pong", 19, GOLD, 700)
    s.text(342, 863, "cp.async K/V[t+1] overlaps QK[t] → softmax[t] → PV[t]; Q remains in shared memory", 16, MUTED, 500)
    s.path("M771 813 V792 H1133 V781", GOLD, 2, True, "7 6")

    s.text(70, 949, "Boundary: prologue/mainloop/epilogue describe data lifetime; pipeline is the overlap inside the mainloop.", 17, MUTED, 500)
    return s.finish()


def timeline_block(s: Svg, x: int, y: int, w: int, h: int, fill: str,
                   stroke: str, top: str, bottom: str = "") -> None:
    s.rect(x, y, w, h, fill, stroke, 8, 1)
    s.text(x + w // 2, y + 25, top, 16, stroke, 700, "middle")
    if bottom:
        s.text(x + w // 2, y + 48, bottom, 14, MUTED, 500, "middle")


def timeline_svg() -> str:
    s = Svg(
        1600, 940,
        "CUTLASS 3.x Ampere runtime pipeline timeline",
        "Timeline comparing the three-stage GEMM cp.async pipeline and the two-stage KV FlashAttention-2 pipeline.",
    )
    s.text(70, 62, "CUTLASS 3.x on Ampere · Pipeline timeline", 34, INK, 750)
    s.text(70, 96, "Conceptual timing, not cycle-scaled; dashed lines mark logical iterations", 19, MUTED, 500)

    # Common x positions: label area then six logical slots.
    left, step, top = 280, 205, 160
    for i in range(7):
        x = left + i * step
        s.line(x, 145, x, 875, GRID, 1, False, "6 8")
        if i < 6:
            s.text(x + step // 2, 138, f"T{i}", 16, MUTED, 600, "middle")

    s.text(70, 176, "GEMM · 3 stages", 25, BLUE, 750)
    s.text(70, 205, "128×128×32", 17, MUTED, 500)
    row_y = [230, 294, 358, 422, 486]
    labels = ["producer / cp.async", "SMEM stage 0", "SMEM stage 1", "SMEM stage 2", "consumer / Tensor Core"]
    for y, label in zip(row_y, labels):
        s.text(70, y + 34, label, 17, INK if "SMEM" not in label else MUTED, 600)

    timeline_block(s, left + 8, row_y[0], step - 16, 52, GOLD_LIGHT, GOLD, "copy A/B K₀")
    timeline_block(s, left + step + 8, row_y[0], step - 16, 52, GOLD_LIGHT, GOLD, "copy A/B K₁")
    timeline_block(s, left + 2 * step + 8, row_y[0], step - 16, 52, GOLD_LIGHT, GOLD, "copy K₂")
    timeline_block(s, left + 3 * step + 8, row_y[0], step - 16, 52, GOLD_LIGHT, GOLD, "copy K₃")
    timeline_block(s, left + 4 * step + 8, row_y[0], step - 16, 52, GOLD_LIGHT, GOLD, "copy K₄")

    for col, stage, tile in ((0, 0, "K₀"), (1, 1, "K₁"), (2, 2, "K₂"), (3, 0, "K₃"), (4, 1, "K₄")):
        timeline_block(s, left + col * step + 8, row_y[1 + stage], step - 16, 52, WHITE, GOLD, f"ready {tile}")
    for col, tile in ((2, "K₀"), (3, "K₁"), (4, "K₂"), (5, "K₃ / drain")):
        timeline_block(s, left + col * step + 8, row_y[4], step - 16, 52, BLUE_LIGHT, BLUE, f"LDSM + MMA {tile}")
    s.path(f"M{left + 2*step - 10} {row_y[0]+58} V{row_y[4]-8}", BLUE, 2, False, "6 5")
    s.text(left + 13, 553, "Prologue: submit Stages−1 copy groups", 16, GOLD, 650)
    s.text(left + 2 * step + 13, 553, "Steady state: copy(next) overlaps consume(current)", 16, BLUE, 650)

    # FA2 timeline.
    s.line(55, 594, 1545, 594, GRID, 2)
    s.text(70, 635, "FA2 · 2-stage KV", 25, PURPLE, 750)
    s.text(70, 664, "64×64 · D=64", 17, MUTED, 500)
    fy = [686, 750, 814]
    flabels = ["producer / cp.async", "stage 0 / stage 1", "consumer chain"]
    for y, label in zip(fy, flabels):
        s.text(70, y + 34, label, 17, INK if "stage" not in label else MUTED, 600)

    for col, tile in ((0, "Q + KV₀"), (1, "KV₁"), (2, "KV₂"), (3, "KV₃")):
        timeline_block(s, left + col * step + 8, fy[0], step - 16, 52, GOLD_LIGHT, GOLD, f"copy {tile}")
    for col, text_value in ((0, "S0 = KV₀"), (1, "S1 = KV₁"), (2, "S0 = KV₂"), (3, "S1 = KV₃")):
        timeline_block(s, left + col * step + 8, fy[1], step - 16, 52, WHITE, GOLD, text_value)
    for col, tile in ((1, "tile 0"), (2, "tile 1"), (3, "tile 2"), (4, "tile 3")):
        timeline_block(s, left + col * step + 8, fy[2], step - 16, 52, PURPLE_LIGHT, PURPLE, f"QK → softmax → PV", tile)
    s.path(f"M{left + step + 15} {fy[0]+58} V{fy[2]-8}", PURPLE, 2, False, "6 5")
    s.text(left + 5 * step + 12, fy[2] + 25, "Epilogue", 17, GREEN, 750)
    s.text(left + 5 * step + 12, fy[2] + 49, "O / l → FP16", 15, MUTED, 500)
    s.text(70, 912, "Synchronization: cp_async_fence commits; cp_async_wait + __syncthreads precede SMEM stage reuse.", 17, MUTED, 500)
    return s.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Directory receiving the two SVG files.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cutlass3_execution_architecture.svg": architecture_svg(),
        "cutlass3_pipeline_timeline.svg": timeline_svg(),
    }
    for filename, content in outputs.items():
        (args.output_dir / filename).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
