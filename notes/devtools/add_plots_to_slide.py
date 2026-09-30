#!/usr/bin/env python3
"""Write the bottom half of pto_einsum_slide.pptx: status text, no charts.

Touches ONLY the bottom half (at or below the y=3.93in divider). Always starts
from the pristine pto_einsum_slide.orig.pptx, so re-running is idempotent and
never stacks edits.

Copy follows the top half's own pattern exactly: a bold lead claim in 1D1D1A
followed by a lighter 3F3F3D elaboration, 9.5pt Microsoft YaHei, and the deck's
existing vocabulary ("generated code" / "hand-written code") rather than internal
tool names — the audience is management.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu, Inches

DECK = Path("/root/workspace/allscan/pto_einsum_slide.pptx")
PRISTINE = DECK.with_suffix(".orig.pptx")
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

# --- geometry (inches): the deck's original bottom-half layout ---------------
PANEL_TOP, PANEL_H = 4.03, 2.82
PROSE_TOP, PROSE_H = 5.02, 1.60
COL_L, COL_R, COL_W = 0.80, 6.94, 5.59

SUBTITLE = "correct on every test we can run — the speed comparison is still being built"
CAPTION = "STATUS · AUGUST 2026"

#: The closing line of the TOP block, promoted from its own text box (id 8) into the
#: bullet list as a final underlined bullet. Its old box and the thin rule that used to
#: separate it (id 7) are removed — the list now grows into that space, so they cannot stay.
TOP_CLOSER = "The work now continues in the compiler stack itself — pto-mlir."

LEFT_BULLETS = [
    ("Runs one attention layer across several chips.",
     " A sequence too long for a single chip is split into pieces and worked on at the same "
     "time, each chip passing its running total to the next while all of them keep computing."),
    ("Built four ways, so the answers can be checked against each other.",
     " A plain reference, a standard multi-chip version, hand-written chip code and generated "
     "chip code — all computing the same thing."),
]
RIGHT_BULLETS = [
    ("Correct on real hardware.",
     " Both directions needed for training now run on one, two and four chips, and all four "
     "versions agree."),
    ("Speed is not settled yet.",
     " On a single chip the generated code is about three times quicker, because it does the "
     "whole job in one visit to the chip instead of three. The multi-chip comparison is held "
     "up by a fault on the hand-written side."),
    ("Faults found here were fixed in the shared tools.",
     " Several real defects surfaced in the underlying compiler and chip libraries and have "
     "gone back to the teams that own them."),
]


def _restyle(run, bold: bool, colour: str, text: str):
    rPr = run.find(f"{A}rPr")
    rPr.set("b", "1" if bold else "0")
    rPr.find(f"{A}solidFill").find(f"{A}srgbClr").set("val", colour)
    run.find(f"{A}t").text = text


def _set_bullets(shape, bullets):
    """Rewrite a bulleted box as bold-lead + elaboration, cloning existing formatting."""
    body = shape.text_frame._txBody
    paras = body.findall(f"{A}p")
    first, rest_tpl = paras[0], (paras[1] if len(paras) > 1 else paras[0])
    for p in paras:
        body.remove(p)
    for i, (lead, tail) in enumerate(bullets):
        p = copy.deepcopy(first if i == 0 else rest_tpl)
        runs = p.findall(f"{A}r")
        for extra in runs[1:]:
            p.remove(extra)
        _restyle(runs[0], True, "1D1D1A", lead)
        tail_run = copy.deepcopy(runs[0])
        _restyle(tail_run, False, "3F3F3D", tail)
        runs[0].addnext(tail_run)
        body.append(p)


def _add_closing_bullet(shape, text):
    """Append one bold, underlined bullet, cloned from the block's own formatting."""
    body = shape.text_frame._txBody
    paras = body.findall(f"{A}p")
    p = copy.deepcopy(paras[-1])                 # carries buChar, indent, spcBef
    runs = p.findall(f"{A}r")
    for extra in runs[1:]:
        p.remove(extra)
    _restyle(runs[0], True, "1D1D1A", text)
    runs[0].find(f"{A}rPr").set("u", "sng")
    body.append(p)


def _set_run(shape, text, para_index=0):
    p = shape.text_frame._txBody.findall(f"{A}p")[para_index]
    p.find(f"{A}r").find(f"{A}t").text = text


def main():
    if not PRISTINE.exists():
        raise SystemExit(f"missing {PRISTINE} — cannot guarantee a clean base")
    shutil.copy2(PRISTINE, DECK)

    prs = Presentation(DECK)
    slide = prs.slides[0]
    by_id = {sh.shape_id: sh for sh in slide.shapes}

    for sh in list(slide.shapes):                      # no charts in this version
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and Emu(sh.top).inches >= PANEL_TOP:
            sh._element.getparent().remove(sh._element)
            print(f"removed chart {sh.name!r}")

    # --- top half: promote the closing line into the bullet list ---
    _add_closing_bullet(by_id[6], TOP_CLOSER)
    for dead in (8, 7):            # the old text box, then the rule that separated it
        sh = by_id[dead]
        sh._element.getparent().remove(sh._element)
        print(f"removed id={dead} ({sh.name})")

    panel = by_id[2]
    panel.top, panel.height = Inches(PANEL_TOP), Inches(PANEL_H)

    _set_run(by_id[53], SUBTITLE, para_index=1)
    _set_run(by_id[56], CAPTION)

    for shape_id, bullets, left in ((54, LEFT_BULLETS, COL_L), (55, RIGHT_BULLETS, COL_R)):
        sh = by_id[shape_id]
        _set_bullets(sh, bullets)
        sh.left, sh.top = Inches(left), Inches(PROSE_TOP)
        sh.width, sh.height = Inches(COL_W), Inches(PROSE_H)
        print(f"id={shape_id}: {len(bullets)} bullets")

    notes = Path("/root/workspace/allscan/devtools/slide_notes.txt")
    if notes.exists():                 # rebuilt from pristine each run, so re-attach them
        slide.notes_slide.notes_text_frame.text = notes.read_text()
        print(f"speaker notes attached ({len(notes.read_text().split())} words)")

    prs.save(DECK)
    print(f"saved {DECK}")


if __name__ == "__main__":
    main()
