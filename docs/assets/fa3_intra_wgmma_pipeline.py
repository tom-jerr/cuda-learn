#!/usr/bin/env python3
"""Generate a paper-style SVG/PDF of the FA3 intra-WGMMA pipeline.

The SVG is the editable source artifact. The PDF is rendered directly from
that SVG through the system librsvg + Cairo libraries, so it stays vector.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import html
from pathlib import Path


W, H = 1500, 850

INK = "#111111"
BLUE = "#2456A6"
BLUE_LIGHT = "#DDE9FA"
QK = "#F3D8A6"
PV = "#CBC4DB"
SOFTMAX = "#E6EEF9"
FLIGHT = "#656565"
WAIT = "#F3F3F3"
MASK = "#D9D9D9"
WHITE = "#FFFFFF"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self) -> None:
        self.out: list[str] = []

    def add(self, raw: str) -> None:
        self.out.append(raw)

    def rect(self, x, y, w, h, fill=WHITE, stroke=INK, sw=1.5,
             dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1.5,
             dash: str | None = None, marker: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if marker:
            extra += f' marker-end="url(#{marker})"'
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def path(self, d, stroke=INK, sw=1.5, fill="none",
             dash: str | None = None, marker: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if marker:
            extra += f' marker-end="url(#{marker})"'
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{extra}/>'
        )

    def text(self, x, y, value, size=18, weight=400, fill=INK,
             anchor="start", italic=False, family="serif",
             baseline: str | None = None) -> None:
        style = ' font-style="italic"' if italic else ""
        base = f' dominant-baseline="{baseline}"' if baseline else ""
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" class="{family}"{style}{base}>'
            f'{esc(value)}</text>'
        )


def box(s: SVG, x, y, w, h, label, fill, sub: str | None = None,
        italic=False, size=18, stroke=INK) -> None:
    s.rect(x, y, w, h, fill, stroke, 1.5)
    if sub:
        s.text(x + w / 2, y + h * 0.40, label, size, 600,
               anchor="middle", italic=italic, baseline="middle")
        s.text(x + w / 2, y + h * 0.73, sub, 13, 400,
               anchor="middle", family="sans", baseline="middle")
    else:
        s.text(x + w / 2, y + h / 2, label, size, 500,
               anchor="middle", italic=italic, baseline="middle")


def axis(s: SVG, y: int, x0=220, x1=1430) -> None:
    s.line(x0, y, x1, y, INK, 1.25, marker="arrow")
    s.text(x1 + 10, y + 5, "time", 16, 400, italic=True)


def dashed_boundary(s: SVG, x: int, y1: int, y2: int) -> None:
    s.line(x, y1, x, y2, INK, 1.2, "6 6")


def double_arrow(s: SVG, x1, x2, y, label, size=16) -> None:
    s.line(x1 + 8, y, x2 - 8, y, BLUE, 1.4)
    s.path(f"M {x1 + 8} {y} L {x1 + 17} {y - 5} M {x1 + 8} {y} L {x1 + 17} {y + 5}",
           BLUE, 1.4)
    s.path(f"M {x2 - 8} {y} L {x2 - 17} {y - 5} M {x2 - 8} {y} L {x2 - 17} {y + 5}",
           BLUE, 1.4)
    s.text((x1 + x2) / 2, y - 9, label, size, 600, BLUE,
           anchor="middle", italic=True)


def panel_a(s: SVG) -> None:
    """Actual cross-consumer steady-state schedule."""
    s.text(18, 42, "(a) Two-consumer steady state", 23, 500, family="sans")
    s.text(42, 103, "consumer 0", 18, 500, family="sans")
    s.text(42, 221, "consumer 1", 18, 500, family="sans")

    y0, y1, bh = 70, 188, 46
    axis(s, 154)
    axis(s, 272)

    # Four deterministic commit slots.
    box(s, 220, y0, 145, bh, "QK(n+1)", QK, "commit group", italic=True)
    box(s, 365, y1, 145, bh, "QK(n+1)", QK, "commit group", italic=True)
    box(s, 510, y0, 145, bh, "PV(n)", PV, "commit group", italic=True)
    box(s, 655, y1, 145, bh, "PV(n)", PV, "commit group", italic=True)

    # wait_group<1> exposes scores; softmax overlaps the younger PV group.
    box(s, 655, y0, 58, bh, "wait", WAIT, "<1>", size=15)
    box(s, 713, y0, 175, bh, "softmax(n+1)", SOFTMAX, "CUDA / SFU", italic=True, size=17)
    box(s, 888, y0, 62, bh, "wait", WAIT, "<0>", size=15)
    box(s, 950, y0, 70, bh, "scale O", WHITE, "× α", italic=True, size=15)

    box(s, 800, y1, 58, bh, "wait", WAIT, "<1>", size=15)
    box(s, 858, y1, 192, bh, "softmax(n+1)", SOFTMAX, "CUDA / SFU", italic=True, size=17)
    box(s, 1050, y1, 62, bh, "wait", WAIT, "<0>", size=15)
    box(s, 1112, y1, 70, bh, "scale O", WHITE, "× α", italic=True, size=15)

    # C0 can resume while C1 is still in softmax.
    box(s, 1020, y0, 145, bh, "QK(n+2)", QK, "next transition", italic=True)
    box(s, 1182, y1, 145, bh, "QK(n+2)", QK, "next transition", italic=True)

    # WGMMA remains in flight below each instruction/CUDA row.
    s.rect(510, 116, 440, 31, FLIGHT, FLIGHT, 0)
    s.text(730, 132, "PV(n) in flight", 16, 600, WHITE,
           anchor="middle", italic=True, baseline="middle")
    s.rect(655, 234, 457, 31, FLIGHT, FLIGHT, 0)
    s.text(883, 250, "PV(n) in flight", 16, 600, WHITE,
           anchor="middle", italic=True, baseline="middle")

    # turn[] hand-offs, aligned to the commit sequence.
    for x in (365, 510, 655, 800, 1020, 1182):
        dashed_boundary(s, x, 56, 285)
    double_arrow(s, 220, 800, 55,
                 "turn[2] token:  C0.QK → C1.QK → C0.PV → C1.PV", 16)
    s.text(1370, 109, "C0 resumes", 14, 600, BLUE, anchor="end", italic=True)
    s.text(1370, 129, "during C1 softmax", 14, 600, BLUE, anchor="end", italic=True)
    s.path("M 1360 134 L 1195 146 L 1030 126", BLUE, 1.3, marker="blue-arrow")


def panel_b(s: SVG) -> None:
    """Explain older/younger WGMMA group semantics once."""
    top = 326
    s.text(18, top + 22, "(b) Two-level pipeline inside each consumer", 23, 500, family="sans")
    y, bh = top + 62, 50
    axis(s, top + 139)

    box(s, 220, y, 190, bh, "QK(next)", QK, "older WGMMA group", italic=True)
    box(s, 410, y, 190, bh, "PV(cur)", PV, "younger WGMMA group", italic=True)
    box(s, 600, y, 105, bh, "wait_group", WAIT, "<1>", size=16)
    box(s, 705, y, 275, bh, "softmax(next)", SOFTMAX, "scores ready · exp2", italic=True)
    box(s, 980, y, 105, bh, "wait_group", WAIT, "<0>", size=16)
    box(s, 1085, y, 150, bh, "stage_empty", BLUE_LIGHT, "arrive", size=16, stroke=BLUE)
    box(s, 1235, y, 150, bh, "rescale O", WHITE, "× alpha(next)", italic=True, size=17)

    s.rect(410, y + bh, 675, 32, FLIGHT, FLIGHT, 0)
    s.text(747, y + bh + 17, "PV(cur) may remain in flight", 16, 600,
           WHITE, anchor="middle", italic=True, baseline="middle")
    dashed_boundary(s, 600, y - 15, y + 103)
    dashed_boundary(s, 1085, y - 15, y + 103)
    double_arrow(s, 600, 1085, y - 16,
                 "CUDA / SFU overlaps the younger PV group", 16)


def panel_c(s: SVG) -> None:
    """Show the causal trip-count invariant protecting stage barriers."""
    top = 574
    s.text(18, top + 22, "(c) Causal mode keeps equal barrier participation", 23, 500, family="sans")
    s.text(42, top + 74, "consumer 0", 18, 500, family="sans")
    s.text(42, top + 137, "consumer 1", 18, 500, family="sans")
    y0, y1, bh = top + 66, top + 129, 38
    axis(s, top + 190)

    x, tw = 220, 150
    for i in range(4):
        label = f"KV tile {i}" if i < 3 else "future tile"
        fill = MASK if i == 3 else WHITE
        box(s, x + i * tw, y0, tw, bh, label, fill, italic=True, size=16)
        box(s, x + i * tw, y1, tw, bh, f"KV tile {i}", WHITE, italic=True, size=16)
    s.text(x + 3 * tw + tw / 2, y0 + 29, "mask = −∞", 12, 600,
           anchor="middle", family="sans")

    end = x + 4 * tw
    dashed_boundary(s, end, top + 52, top + 211)
    double_arrow(s, x, end, top + 51, "same number of KV tiles", 16)
    box(s, end + 70, top + 92, 245, 48, "stage_empty[stage]", BLUE_LIGHT,
        "2 arrivals → phase flips", size=17, stroke=BLUE)
    s.path(f"M {end} {y0 + bh / 2} L {end + 42} {y0 + bh / 2} L {end + 42} {top + 116} L {end + 70} {top + 116}",
           BLUE, 1.5, marker="blue-arrow")
    s.path(f"M {end} {y1 + bh / 2} L {end + 42} {y1 + bh / 2} L {end + 42} {top + 116}",
           BLUE, 1.5)

    # Compact legend.
    ly = 825
    items = [(220, QK, "WGMMA QK"), (420, PV, "WGMMA PV"),
             (620, SOFTMAX, "softmax / SFU"), (850, FLIGHT, "WGMMA in flight"),
             (1100, BLUE_LIGHT, "mbarrier event")]
    for ix, fill, label in items:
        s.rect(ix, ly - 12, 28, 14, fill, INK if fill != BLUE_LIGHT else BLUE, 1)
        s.text(ix + 38, ly, label, 14, 400, family="sans")


def build_svg() -> str:
    s = SVG()
    s.add(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">Two-consumer intra-WGMMA and softmax pipeline</title>
<desc id="desc">Paper-style timing diagram showing deterministic turn-token WGMMA commits, softmax overlap with an in-flight PV group, and equal causal barrier participation.</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>
  <marker id="blue-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker>
  <style>
    .serif {{ font-family: "Times New Roman", Times, "Droid Sans Fallback", "Noto Sans CJK SC", serif; }}
    .sans {{ font-family: Arial, Helvetica, "Droid Sans Fallback", "Noto Sans CJK SC", sans-serif; }}
    text {{ text-rendering: geometricPrecision; }}
  </style>
</defs>''')
    s.rect(0, 0, W, H, WHITE, "none", 0)
    panel_a(s)
    panel_b(s)
    panel_c(s)
    s.add("</svg>")
    return "\n".join(s.out) + "\n"


class RsvgRectangle(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double),
                ("width", ctypes.c_double), ("height", ctypes.c_double)]


def render_pdf(svg_path: Path, pdf_path: Path) -> None:
    """Render SVG to vector PDF with system librsvg and Cairo."""
    rsvg_name = ctypes.util.find_library("rsvg-2")
    cairo_name = ctypes.util.find_library("cairo")
    gobject_name = ctypes.util.find_library("gobject-2.0")
    if not (rsvg_name and cairo_name and gobject_name):
        raise RuntimeError("librsvg, Cairo, and GObject are required for PDF output")

    rsvg = ctypes.CDLL(rsvg_name)
    cairo = ctypes.CDLL(cairo_name)
    gobject = ctypes.CDLL(gobject_name)

    rsvg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    rsvg.rsvg_handle_new_from_file.restype = ctypes.c_void_p
    rsvg.rsvg_handle_render_document.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.POINTER(RsvgRectangle),
                                                  ctypes.POINTER(ctypes.c_void_p)]
    rsvg.rsvg_handle_render_document.restype = ctypes.c_bool
    cairo.cairo_pdf_surface_create.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
    cairo.cairo_pdf_surface_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = [ctypes.c_void_p]
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_finish.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
    gobject.g_object_unref.argtypes = [ctypes.c_void_p]

    error = ctypes.c_void_p()
    handle = rsvg.rsvg_handle_new_from_file(str(svg_path).encode(), ctypes.byref(error))
    if not handle:
        raise RuntimeError(f"librsvg could not open {svg_path}")

    # 750 × 410 pt: compact, paper-friendly landscape output.
    pdf_w, pdf_h = W * 0.5, H * 0.5
    surface = cairo.cairo_pdf_surface_create(str(pdf_path).encode(), pdf_w, pdf_h)
    cr = cairo.cairo_create(surface)
    viewport = RsvgRectangle(0, 0, pdf_w, pdf_h)
    ok = rsvg.rsvg_handle_render_document(handle, cr, ctypes.byref(viewport), ctypes.byref(error))
    cairo.cairo_destroy(cr)
    cairo.cairo_surface_finish(surface)
    cairo.cairo_surface_destroy(surface)
    gobject.g_object_unref(handle)
    if not ok:
        raise RuntimeError(f"librsvg failed while rendering {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", type=Path,
        default=Path(__file__).with_suffix(".svg"),
        help="output .svg or .pdf path; the sibling vector format is also generated",
    )
    args = parser.parse_args()
    stem = args.output.with_suffix("")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(build_svg(), encoding="utf-8")
    render_pdf(svg_path, pdf_path)


if __name__ == "__main__":
    main()
