#!/usr/bin/env python3
"""Editable R2S ownership diagram. All offsets are half-element slots, not physical registers."""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
import html
from pathlib import Path

W,H=1200,1150
INK='#172033'; MUTED='#5D687A'; GRID='#CBD3DE'; PANEL='#F7F9FC'
BLUE='#174EA6'; BLUE_LIGHT='#DCE8FF'; GOLD='#995600'; GOLD_LIGHT='#FCE9C8'

def build_svg():
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
         '<title id="title">CuTe R2S retile：C 坐标与线程私有槽位的对应</title>',
         '<desc id="desc">固定线程37，宏块im=in=0。C跨8行对应槽位加2，跨16列对应槽位加16。R2S使用32-bit UniversalCopy，与x4 ldmatrix无关。</desc>',
         '<style>text{font-family:Arial,"Droid Sans Fallback","Noto Sans CJK SC",sans-serif}.mono{font-family:"DejaVu Sans Mono","Droid Sans Fallback",monospace}</style>',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#5D687A"/></marker></defs>']
    def rect(x,y,w,h,fill='white',stroke=GRID,sw=1):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(x,y,value,size=19,color=INK,bold=False,anchor='start',mono=False):
        cls=' class="mono"' if mono else ''
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}"{cls}>{html.escape(str(value))}</text>')
    def line(x1,y1,x2,y2,arrow=False):
        markers=' marker-start="url(#arrow)" marker-end="url(#arrow)"' if arrow else ''
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.5"{markers}/>')
    rect(0,0,W,H,'white','white')
    text(40,46,'R2S retile：矩阵坐标 ≠ 寄存器槽位',30,bold=True)
    text(40,79,'R2S 使用 32-bit copy；stride 的单位是 half 槽位，与 x4 ldmatrix 无关。',19,MUTED)
    text(40,122,'((2,(2,2)),4,4):((1,(2,16)),4,32)',29,mono=True)
    text(40,150,'  ((a,(h,z)),im,in)      对应 slot = a + 2h + 16z + 4im + 32in',19,mono=True)
    widths=[145,115,215,305,340]; xs=[40]
    for w in widths: xs.append(xs[-1]+w)
    rows=[['坐标','大小','本线程 slot 增量','C 的逻辑坐标变化','含义'],
          ['a','2','+1','列 +1','一条 32-bit copy 内的两个 half'],
          ['h','2','+2','行 +8','MMA atom 内的上下两组行'],
          ['z','2','+16','列 +16','同一 32×32 宏块内的两个 N 子块'],
          ['im','4','+4','行 +32','下一个 M 宏块'],
          ['in','4','+32','列 +32','下一个 N 宏块']]
    for ri,row in enumerate(rows):
        y=169+ri*35
        for ci,value in enumerate(row):
            fill=PANEL if ri==0 else BLUE_LIGHT if ri==2 else GOLD_LIGHT if ri==3 else 'white'
            rect(xs[ci],y,widths[ci],35,fill)
            text(xs[ci]+12,y+24,value,17,bold=ri==0)
    text(40,425,'同一组 8 个值：线程 37，im = in = 0',24,bold=True)
    text(40,456,'左：32×32 宏块中的 C 坐标',21,bold=True)
    text(600,456,'右：该线程原 accumulator 的槽位',21,bold=True)
    gx,gy,cell=86,507,12
    selected={}
    for u in range(8):
        a=u%2;h=(u//2)%2;z=u//4
        r,c=17+8*h,2+a+16*z
        slot=a+2*h+16*z
        selected[(r,c)]=(u,slot)
        assert slot in [0,1,2,3,16,17,18,19]
    for r in range(32):
        for c in range(32):
            fill='white';stroke='#E2E7EE'
            if (r,c) in selected:
                u,_=selected[(r,c)];fill=BLUE_LIGHT if u<4 else GOLD_LIGHT;stroke=BLUE if u<4 else GOLD
            rect(gx+c*cell,gy+r*cell,cell,cell,fill,stroke)
    for c in [0,3,19,31]:text(gx+(c+.5)*cell,gy-8,c,14,MUTED,anchor='middle')
    for r in [0,17,25,31]:text(gx-11,gy+(r+.5)*cell+5,r,15,MUTED,anchor='end')
    line(gx+3*cell,gy-29,gx+19*cell,gy-29,True)
    text(gx+11*cell,gy-37,'z：列 +16',16,GOLD,anchor='middle')
    line(48,gy+17.5*cell,48,gy+25.5*cell,True)
    text(41,gy+17.5*cell-12,'h',17,BLUE,anchor='middle')
    text(40,922,'h：跨 8 行；z：跨 16 列。',20,bold=True)
    text(40,949,'图中是逻辑坐标，尚未套用 shared swizzle。',17,MUTED)

    rx,ry,cw,ch=600,489,66,66
    for slot in range(32):
        x=rx+(slot%8)*cw;y=ry+(slot//8)*ch
        active=slot in [0,1,2,3,16,17,18,19]
        fill=BLUE_LIGHT if slot<4 else GOLD_LIGHT if 16<=slot<20 else PANEL
        rect(x,y,cw,ch,fill)
        text(x+cw/2,y+28,slot,21,INK if active else MUTED,True,anchor='middle',mono=True)
        if active:
            u=slot if slot<4 else slot-12
            text(x+cw/2,y+52,f'u={u}',16,BLUE if slot<4 else GOLD,anchor='middle',mono=True)
    text(600,780,'灰格属于其他宏块；这里只画槽位 0..31。',17,MUTED)
    text(600,807,'数字是 half 槽位，不是机器寄存器 R 编号。',17,MUTED)
    pairs=[('u=0,1','C(17,{2,3})','slot 0,1',BLUE),
           ('u=2,3','C(25,{2,3})','slot 2,3',BLUE),
           ('u=4,5','C(17,{18,19})','slot 16,17',GOLD),
           ('u=6,7','C(25,{18,19})','slot 18,19',GOLD)]
    for i,(u,c,s,col) in enumerate(pairs):
        y=844+i*30
        text(600,y,u,18,col,mono=True);text(710,y,c,18,col,mono=True);text(935,y,s,18,col,mono=True)
    line(40,974,1160,974)
    text(40,1010,'从原布局直接代入，就得到 stride 16：',21,bold=True)
    text(40,1045,'原 C：slot = v + 4*im + 16*j',23,mono=True)
    text(40,1080,'retile：v = a + 2*h，j = 2*in + z',23,mono=True)
    text(40,1115,'所以：slot = a + 2*h + 16*z + 4*im + 32*in',23,mono=True)
    out.append('</svg>')
    return '\n'.join(out)+'\n'

class RsvgRectangle(ctypes.Structure):
    _fields_=[('x',ctypes.c_double),('y',ctypes.c_double),('width',ctypes.c_double),('height',ctypes.c_double)]

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



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=Path(__file__).with_suffix('.svg'))
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(build_svg(),encoding='utf-8')
    render_png(args.output,args.output.with_suffix('.png'))

if __name__=='__main__':
    main()
