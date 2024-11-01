#!/usr/bin/env python3

import argparse
import math
from typing import List

from stability_paper.tools import surface_code_tiles


def pentagonal_surface_code_svg(*, diam: int, show_order: bool = False, code_type: str) -> str:
    if code_type == 'memory':
        tiles = surface_code_tiles(diam=diam, top_bot_basis='X', side_basis='Z', x_order=[0, 1j, 1, 1 + 1j], z_order=[0, 1, 1j, 1 + 1j])
    elif code_type == 'stability_x':
        tiles = surface_code_tiles(diam=diam, top_bot_basis='X', side_basis='X', x_order=[0, 1j, 1, 1 + 1j], z_order=[0, 1, 1j, 1 + 1j])
    elif code_type == 'stability_z':
        tiles = surface_code_tiles(diam=diam, top_bot_basis='Z', side_basis='Z', x_order=[0, 1j, 1, 1 + 1j], z_order=[0, 1, 1j, 1 + 1j])
    else:
        raise NotImplementedError(f'{code_type=}')
    lines = []
    canvas_width = 512
    canvas_height = 512
    used_set = {q for tile in tiles for q in tile.used_set}
    measure_set = {tile.measure_qubit for tile in tiles}
    data_set = {q for tile in tiles for q in tile.data_set}
    min_r = min(q.real for q in used_set)
    min_i = min(q.imag for q in used_set)
    max_r = max(q.real for q in used_set)
    max_i = max(q.imag for q in used_set)
    min_c = min_r + min_i*1j
    max_c = max_r + max_i*1j
    pad = 5
    min_c -= (1 + 1j) * pad
    max_c += (1 + 1j) * pad
    scale = max((max_c.real - min_c.real) / canvas_width, (max_c.imag - min_c.imag) / canvas_height)

    BACKGROUND_X = '#FF0000'
    BACKGROUND_Z = '#0000FF'
    BACKGROUND_STROKE = '#000000'
    QUBIT_RADIUS = 0.1 / scale
    stroke_width = 0.03 / scale

    def pt0(c: complex) -> complex:
        c -= min_c
        c /= scale
        return c

    def dt0(d: complex) -> complex:
        return pt0(d) - pt0(0)

    def pt(c: complex) -> str:
        c = pt0(c)
        return f'{c.real},{c.imag}'

    def dt(d: complex) -> str:
        d = pt0(d) - pt0(0)
        return f'{d.real},{d.imag}'

    lines.append(f"""<svg viewBox="0 0 {canvas_width} {canvas_height}" xmlns="http://www.w3.org/2000/svg">""")
    for tile in tiles:
        background = BACKGROUND_X if tile.basis == 'X' else BACKGROUND_Z

        if len(tile.data_set) == 2:
            a, b = tile.data_set
            da = a - tile.measure_qubit
            db = b - tile.measure_qubit
            dab = math.atan2(da.imag, da.real) - math.atan2(db.imag, db.real)
            dab %= math.pi * 2
            if dab < math.pi:
                a, b = b, a
            lines.append(f'<path d="M{pt(a)} a1,1 0 0,0 {dt(b - a)} M {pt(a)} {pt(b)}" fill="{background}" stroke="{BACKGROUND_STROKE}" stroke-width="{stroke_width}" />')
        else:
            qs = sorted(tile.data_set, key=lambda p2: math.atan2(p2.imag - tile.measure_qubit.imag, p2.real - tile.measure_qubit.real))
            x = f'<path d="M{pt(qs[-1])}'
            for q in qs:
                x += ' ' + pt(q)
            x += '"'
            lines.append(f'{x} fill="{background}" stroke="{BACKGROUND_STROKE}"  stroke-width="{stroke_width}" />')
    for q in measure_set:
        lines.append(f'<circle cx="{pt0(q).real}" cy="{pt0(q).imag}" r="{QUBIT_RADIUS}" fill="black" stroke="black" stroke-width="{stroke_width}" />')
    for q in data_set:
        lines.append(f'<circle cx="{pt0(q).real}" cy="{pt0(q).imag}" r="{QUBIT_RADIUS}" fill="white" stroke="black" stroke-width="{stroke_width}" />')

    if show_order:
        for e in tiles:
            c = e.measure_qubit
            if len(e.data_set) == 3:
                c = 0
                for q in e.data_set:
                    c += q
                c /= len(e.data_set)
            pts: List[complex] = []

            x = f'<path d="M'
            arrow_color = "white"
            delay = 0
            prev = None
            for q in e.ordered_data:
                if q is not None:
                    v = q * 0.6 + c * 0.4
                    x += pt(v) + ' '
                    v = pt0(v)
                    pts.append(v)
                    for d in range(delay):
                        if prev is None:
                            prev = v
                        v2 = (prev + v) / 2
                        lines.append(
                            f'<circle cx="{v2.real}" cy="{v2.imag}" r="{dt0(d * 0.06 + 0.04).real}" stroke="yellow" stroke-width="{stroke_width}" fill="none" />')
                    delay = 0
                    prev = v
                else:
                    delay += 1
            x = x.strip()
            x += f'" fill="none" stroke="{arrow_color}" stroke-width="{stroke_width}" />'
            lines.append(x)

            # Draw arrow at end of arrow.
            if len(pts) > 1:
                p = pts[-1]
                d2 = p - pts[-2]
                if d2:
                    d2 /= abs(d2)
                    d2 *= 0.1 / scale
                a = p + d2
                b = p + d2 * 1j
                c = p + d2 * -1j
                lines.append(
                    f'<path'
                    f' d="M{a.real},{a.imag} {b.real},{b.imag} {c.real},{c.imag} {a.real},{a.imag}"'
                    f' stroke="none"'
                    f' fill="{arrow_color}" />'
                )

    lines.append("</svg>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str)
    parser.add_argument("--diam", required=True, type=int)
    parser.add_argument('--show_order', action='store_true')
    parser.add_argument('--type', choices=['stability_x', 'stability_z', 'memory'])
    args = parser.parse_args()
    svg = pentagonal_surface_code_svg(diam=args.diam, show_order=args.show_order, code_type=args.type)
    if not args.out:
        print(svg)
    else:
        with open(args.out, 'w') as f:
            print(svg, file=f)


if __name__ == '__main__':
    main()
