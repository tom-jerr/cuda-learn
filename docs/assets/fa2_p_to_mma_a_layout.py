#!/usr/bin/env python3
"""Generate an editable SVG explaining FA2's P(C-registers) -> MMA-A layout.

The concrete example uses lane 5, so g=1 and t=1.  The geometry and labels
are deterministic and deliberately mirror the PTX m16n8k16 fragment ABI.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import html
from pathlib import Path


W, H = 1600, 1030

INK = "#172033"
MUTED = "#5D687A"
GRID = "#AEB8C7"
PANEL = "#F7F9FC"
WHITE = "#FFFFFF"
BLUE = "#2F6FED"
BLUE_LIGHT = "#DCE8FF"
BLUE_DARK = "#174EA6"
GOLD = "#D88916"
GOLD_LIGHT = "#FCE9C8"
GREEN = "#17845B"
GREEN_LIGHT = "#DDF4EA"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(self, x, y, w, h, fill=WHITE, stroke=GRID, sw=1,
             rx=0, dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1.5,
             marker: str | None = None, dash: str | None = None) -> None:
        extra = f' marker-end="url(#{marker})"' if marker else ""
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def path(self, d, stroke=INK, sw=1.5, fill="none",
             marker: str | None = None) -> None:
        extra = f' marker-end="url(#{marker})"' if marker else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{extra}/>'
        )

    def text(self, x, y, value, size=18, weight=400, fill=INK,
             anchor="start", family="sans", baseline: str | None = None) -> None:
        base = f' dominant-baseline="{baseline}"' if baseline else ""
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" class="{family}"{base}>'
            f'{esc(value)}</text>'
        )


def label_box(s: SVG, x, y, w, h, label, fill, stroke, size=17,
              text_fill=INK, weight=600) -> None:
    s.rect(x, y, w, h, fill, stroke, 1.4, 7)
    s.text(x + w / 2, y + h / 2 + 1, label, size, weight, text_fill,
           anchor="middle", family="mono", baseline="middle")


def matrix_grid(s: SVG, x: int, y: int, rows: int, cols: int, cell: int,
                title: str, col_offset: int,
                highlights: dict[tuple[int, int], tuple[str, str, str]]) -> None:
    """Draw a matrix with (label, fill, text-color) highlighted cells."""
    s.text(x + cols * cell / 2, y - 42, title, 19, 700, INK,
           anchor="middle")
    s.text(x + cols * cell / 2, y - 17,
           f"P[:, {col_offset}:{col_offset + cols}]", 15, 500, MUTED,
           anchor="middle", family="mono")
    for c in range(cols):
        s.text(x + c * cell + cell / 2, y - 5, str(col_offset + c), 11, 500,
               MUTED, anchor="middle", family="mono")
    for r in range(rows):
        s.text(x - 9, y + r * cell + cell / 2, str(r), 11, 500, MUTED,
               anchor="end", family="mono", baseline="middle")
        for c in range(cols):
            key = (r, c)
            if key in highlights:
                label, fill, text_fill = highlights[key]
                s.rect(x + c * cell, y + r * cell, cell, cell,
                       fill, text_fill, 1.5)
                s.text(x + c * cell + cell / 2,
                       y + r * cell + cell / 2 + 1,
                       label, 12, 700, text_fill, anchor="middle",
                       family="mono", baseline="middle")
            else:
                s.rect(x + c * cell, y + r * cell, cell, cell,
                       WHITE, GRID, 0.65)


def build_svg() -> str:
    s = SVG()
    s.add(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">FA2 中 P 从 QK accumulator layout 到 PV MMA A layout</title>
<desc id="desc">以 lane 5 为例，展示两个相邻的 16×8 QK accumulator fragments 如何在不跨 lane 搬运数据的情况下组成一个 16×16 FP16 MMA A fragment。</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>
  <marker id="blue-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker>
  <style>
    .sans {{ font-family: Arial, Helvetica, "Droid Sans Fallback", "Noto Sans CJK SC", sans-serif; }}
    .mono {{ font-family: "DejaVu Sans Mono", "Droid Sans Fallback", "Noto Sans Mono CJK SC", monospace; }}
    text {{ text-rendering: geometricPrecision; }}
  </style>
</defs>''')
    s.rect(0, 0, W, H, WHITE, "none", 0)

    # Title and invariant.
    s.text(48, 48, "P 为什么能零 shuffle 变成第二次 MMA 的 A fragment？",
           29, 700)
    s.text(48, 78,
           "核心：QK 的两个相邻 C(16×8) fragment，在同一个 lane 内恰好按 PV 的 A(16×16) ABI 顺序排列。",
           17, 400, MUTED)

    s.rect(48, 103, 1504, 76, PANEL, GRID, 1.2, 10)
    s.text(72, 132, "通用 lane 分解", 16, 700, MUTED)
    label_box(s, 72, 143, 160, 27, "g = lane >> 2", BLUE_LIGHT, BLUE, 14,
              BLUE_DARK)
    label_box(s, 244, 143, 160, 27, "t = lane & 3", GOLD_LIGHT, GOLD, 14,
              "#8C5700")
    s.text(438, 158,
           "本图取 lane=5 → g=1, t=1；因此这个 lane 负责 row 1 / 9，以及每个 8-column fragment 中的 col 2 / 3。",
           16, 500, INK, baseline="middle")

    # Main panels.
    s.rect(48, 204, 638, 548, PANEL, GRID, 1.2, 12)
    s.text(72, 238, "① QK：两个相邻的 C fragments", 21, 700)
    s.text(72, 265, "同一个 lane 的 c0…c3；第二块只把全局列号加 8",
           15, 400, MUTED)

    c0 = {
        (1, 2): ("c0", BLUE_LIGHT, BLUE_DARK),
        (1, 3): ("c1", BLUE_LIGHT, BLUE_DARK),
        (9, 2): ("c2", BLUE_LIGHT, BLUE_DARK),
        (9, 3): ("c3", BLUE_LIGHT, BLUE_DARK),
    }
    c1 = {
        (1, 2): ("c0′", GOLD_LIGHT, "#8C5700"),
        (1, 3): ("c1′", GOLD_LIGHT, "#8C5700"),
        (9, 2): ("c2′", GOLD_LIGHT, "#8C5700"),
        (9, 3): ("c3′", GOLD_LIGHT, "#8C5700"),
    }
    matrix_grid(s, 92, 334, 16, 8, 24, "C fragment j=0", 0, c0)
    matrix_grid(s, 390, 334, 16, 8, 24, "C fragment j=1", 8, c1)

    # Compact coordinate captions under C fragments.
    s.text(188, 737, "c = P[(1,9), (2,3)]", 13, 600, BLUE_DARK,
           anchor="middle", family="mono")
    s.text(486, 737, "c′ = P[(1,9), (10,11)]", 13, 600, "#8C5700",
           anchor="middle", family="mono")

    # Middle conversion panel.
    s.rect(714, 204, 330, 548, WHITE, GRID, 1.2, 12)
    s.text(879, 238, "② 只改类型和解释方式", 21, 700, anchor="middle")
    s.text(879, 265, "没有跨 lane 数据交换", 15, 600, GREEN,
           anchor="middle")

    label_box(s, 757, 305, 244, 42, "C[j]  = c0 c1 c2 c3",
              BLUE_LIGHT, BLUE, 15, BLUE_DARK)
    label_box(s, 757, 363, 244, 42, "C[j+1]= c0′ c1′ c2′ c3′",
              GOLD_LIGHT, GOLD, 15, "#8C5700")
    s.line(879, 420, 879, 467, GREEN, 2.2, "blue-arrow")
    s.text(899, 445, "FP32 → FP16", 14, 700, GREEN, baseline="middle")

    labels = [
        ("p0 = pack(c0,  c1)", BLUE_LIGHT, BLUE, BLUE_DARK),
        ("p1 = pack(c2,  c3)", BLUE_LIGHT, BLUE, BLUE_DARK),
        ("p2 = pack(c0′, c1′)", GOLD_LIGHT, GOLD, "#8C5700"),
        ("p3 = pack(c2′, c3′)", GOLD_LIGHT, GOLD, "#8C5700"),
    ]
    for i, (label, fill, stroke, text_fill) in enumerate(labels):
        label_box(s, 757, 480 + i * 54, 244, 40, label, fill, stroke,
                  14, text_fill)
    s.text(879, 711, "四个 uint32 = 八个 FP16 A values",
           14, 600, MUTED, anchor="middle")

    # Right A matrix panel.
    s.rect(1072, 204, 480, 548, PANEL, GRID, 1.2, 12)
    s.text(1096, 238, "③ PV：一个 A fragment (16×16)", 21, 700)
    s.text(1096, 265, "A 的 a0…a7 就是左边八个值，顺序完全一致",
           15, 400, MUTED)
    ah = {
        (1, 2): ("a0", BLUE_LIGHT, BLUE_DARK),
        (1, 3): ("a1", BLUE_LIGHT, BLUE_DARK),
        (9, 2): ("a2", BLUE_LIGHT, BLUE_DARK),
        (9, 3): ("a3", BLUE_LIGHT, BLUE_DARK),
        (1, 10): ("a4", GOLD_LIGHT, "#8C5700"),
        (1, 11): ("a5", GOLD_LIGHT, "#8C5700"),
        (9, 10): ("a6", GOLD_LIGHT, "#8C5700"),
        (9, 11): ("a7", GOLD_LIGHT, "#8C5700"),
    }
    matrix_grid(s, 1118, 334, 16, 16, 24, "PV operand A", 0, ah)
    s.text(1310, 737, "A = P[:, 0:16]", 13, 600, GREEN,
           anchor="middle", family="mono")

    # Structural arrows between panels.
    s.path("M 686 474 L 714 474", INK, 2, marker="arrow")
    s.path("M 1044 474 L 1072 474", INK, 2, marker="arrow")

    # Bottom proof / mapping strip.
    s.rect(48, 780, 1504, 202, WHITE, GRID, 1.2, 12)
    s.text(72, 815, "一行看懂这个等价关系", 21, 700)
    s.text(72, 849, "QK C[j]", 15, 700, BLUE_DARK, family="mono")
    s.text(188, 849, "= [c0,c1,c2,c3]", 15, 500, INK, family="mono")
    s.text(418, 849, "→", 20, 700, MUTED, family="mono")
    s.text(455, 849, "PV A 的 [a0,a1,a2,a3]", 15, 700, BLUE_DARK,
           family="mono")
    s.text(72, 882, "QK C[j+1]", 15, 700, "#8C5700", family="mono")
    s.text(188, 882, "= [c0′,c1′,c2′,c3′]", 15, 500, INK,
           family="mono")
    s.text(418, 882, "→", 20, 700, MUTED, family="mono")
    s.text(455, 882, "PV A 的 [a4,a5,a6,a7]", 15, 700, "#8C5700",
           family="mono")

    s.line(805, 827, 805, 941, GRID, 1.2)
    s.text(837, 849, "同一 lane", 16, 700, GREEN)
    s.text(937, 849, "同一组 row：g 与 g+8", 16, 500, INK)
    s.text(837, 882, "相邻 fragment", 16, 700, GREEN)
    s.text(967, 882, "列区间从 [0,8) 接到 [8,16)", 16, 500, INK)
    s.text(837, 915, "所以", 16, 700, GREEN)
    s.text(890, 915, "只需 pack；不需要 shared transpose 或 warp shuffle。",
           16, 600, INK)
    s.text(72, 956,
           "推广到 pk=16h：把 QK 的 fragment pn=2h 与 pn+1 拼接，即得到 P[:, pk:pk+16] 的 MMA-A fragment。",
           15, 600, MUTED, family="mono")

    s.add("</svg>")
    return "\n".join(s.parts) + "\n"


class RsvgRectangle(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double),
                ("width", ctypes.c_double), ("height", ctypes.c_double)]


def render_png(svg_path: Path, png_path: Path) -> None:
    """Render the SVG through system librsvg and Cairo for visual review."""
    rsvg_name = ctypes.util.find_library("rsvg-2")
    cairo_name = ctypes.util.find_library("cairo")
    gobject_name = ctypes.util.find_library("gobject-2.0")
    if not (rsvg_name and cairo_name and gobject_name):
        raise RuntimeError("librsvg, Cairo, and GObject are required")

    rsvg = ctypes.CDLL(rsvg_name)
    cairo = ctypes.CDLL(cairo_name)
    gobject = ctypes.CDLL(gobject_name)
    rsvg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p,
                                                ctypes.POINTER(ctypes.c_void_p)]
    rsvg.rsvg_handle_new_from_file.restype = ctypes.c_void_p
    rsvg.rsvg_handle_render_document.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RsvgRectangle),
        ctypes.POINTER(ctypes.c_void_p)]
    rsvg.rsvg_handle_render_document.restype = ctypes.c_bool
    cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int]
    cairo.cairo_image_surface_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = [ctypes.c_void_p]
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_char_p]
    cairo.cairo_surface_write_to_png.restype = ctypes.c_int
    cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
    gobject.g_object_unref.argtypes = [ctypes.c_void_p]

    error = ctypes.c_void_p()
    handle = rsvg.rsvg_handle_new_from_file(str(svg_path).encode(),
                                             ctypes.byref(error))
    if not handle:
        raise RuntimeError(f"librsvg could not open {svg_path}")
    surface = cairo.cairo_image_surface_create(0, W, H)  # CAIRO_FORMAT_ARGB32
    cr = cairo.cairo_create(surface)
    viewport = RsvgRectangle(0, 0, W, H)
    ok = rsvg.rsvg_handle_render_document(handle, cr, ctypes.byref(viewport),
                                           ctypes.byref(error))
    cairo.cairo_destroy(cr)
    status = cairo.cairo_surface_write_to_png(surface, str(png_path).encode())
    cairo.cairo_surface_destroy(surface)
    gobject.g_object_unref(handle)
    if not ok or status != 0:
        raise RuntimeError(f"failed to render {svg_path} to PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", type=Path,
        default=Path(__file__).with_suffix(".svg"),
        help="output SVG path; a sibling PNG preview is also generated",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    render_png(args.output, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
