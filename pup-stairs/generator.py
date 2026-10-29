#!/usr/bin/env python3
"""PUP-STAIRS parametric part generator.

Parts:
  PLY_GAUGE  - 7-slot comb: which slot slides snug onto a panel edge = true
               ply thickness (no vernier reading). Workshop tool, PLA OK.
  FLANGE_90  - corner glue-rivet bracket: hot glue through the holes forms
               mechanical keys; converts butt joints to shear joints. ABS.

Usage:  python generator.py [ALL|PLY_GAUGE|FLANGE_90]
Exports land in ./stl/. Every export is reloaded from disk and gated.
"""
import os
import sys

import numpy as np
import trimesh

# =============== PARAMS ===============
PART = "ALL"                 # overridden by argv[1]

# --- PLY_GAUGE ---
SLOTS = [5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2]  # mm; 1/4in sanded ply actual
                                             # runs ~5.2, gauge brackets it
FINGER_W = 8.0               # between slots
END_W = 10.0                 # end fingers
SLOT_DEPTH = 15.0            # slot reach from bottom edge
GAUGE_H = 30.0               # plate height
GAUGE_T = 5.0                # plate thickness (stiff enough to push on)
TAB_W = 8.0                  # ear tab marks the SMALL (5.0) end
TAB_H = 4.0                  # proud of top edge
OVERLAP = 1.0                # cutter break-through margin (rule 5)
EMBED = 1.0                  # additive-feature embedment into parent

# --- FLANGE_90 ---
LEG_LEN = 60.0               # along the corner
LEG_DEPTH = 40.0             # each leg's reach onto its panel
LEG_T = 3.2                  # 8 perimeters at 0.4 nozzle
GUSSET_R = 27.0              # gusset triangle leg (45 deg -> printable)
GUSSET_T = 3.2
HOLE_D = 6.0                 # glue-rivet ports
HOLE_X = [12.0, 22.0, 38.0, 48.0]  # clear of gusset slabs (ends + middle)
HOLE_POS = 22.0              # hole row: dist from corner along leg depth
# ======================================

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl")


def box(sx, sy, sz, at=(0, 0, 0)):
    """Axis-aligned box with MIN corner at `at` (easier bookkeeping)."""
    b = trimesh.creation.box(extents=[sx, sy, sz])
    b.apply_translation([at[0] + sx / 2, at[1] + sy / 2, at[2] + sz / 2])
    return b


def weld(m):
    """Conservative pre-export cleanup per boolean-pitfalls rule 8."""
    raw = m.copy()
    m.vertices = np.round(m.vertices, 3)
    m.merge_vertices()
    keep = np.array([len(set(f)) == 3 for f in m.faces])
    m.update_faces(keep)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        parts = sorted(parts, key=lambda p: p.volume)
        m = parts[-1]
    if not m.is_watertight:
        return raw
    return m


def union(meshes):
    return trimesh.boolean.union(meshes, engine="manifold")


def difference(a, cutters):
    return trimesh.boolean.difference([a] + cutters, engine="manifold")


# ---------------- PLY_GAUGE ----------------

def build_ply_gauge():
    length = 2 * END_W + sum(SLOTS) + (len(SLOTS) - 1) * FINGER_W
    plate = box(length, GAUGE_H, GAUGE_T)
    # ear tab marks the small end (additive marker, rule 6)
    tab = box(TAB_W, TAB_H + EMBED, GAUGE_T, at=(0, GAUGE_H - EMBED, 0))
    body = union([plate, tab])
    # slot cutters from the bottom edge, break-through both below and in Z
    cutters = []
    x = END_W
    for w in SLOTS:
        cutters.append(box(w, SLOT_DEPTH + OVERLAP, GAUGE_T + 2 * OVERLAP,
                           at=(x, -OVERLAP, -OVERLAP)))
        x += w + FINGER_W
    m = weld(difference(body, cutters))
    bbox = (length, GAUGE_H + TAB_H, GAUGE_T)
    return m, bbox, 0  # genus 0: slots are open notches


# ---------------- FLANGE_90 ----------------

def build_flange_90():
    leg_a = box(LEG_LEN, LEG_DEPTH, LEG_T)                # on the bed
    leg_b = box(LEG_LEN, LEG_T, LEG_DEPTH)                # vertical
    # gussets: right triangles in YZ, seated 0.3 into both legs (rule 3)
    prof = np.array([[LEG_T - 0.3, LEG_T - 0.3],
                     [GUSSET_R, LEG_T - 0.3],
                     [LEG_T - 0.3, GUSSET_R]])
    poly = trimesh.path.polygons.Polygon(prof)
    slabs = []
    for x0 in (0.0, (LEG_LEN - GUSSET_T) / 2, LEG_LEN - GUSSET_T):
        g = trimesh.creation.extrude_polygon(poly, GUSSET_T)
        # extrude is along +Z of the profile plane; profile is (y,z) ->
        # rotate so extrusion runs along +X: map (px,py,pz)->(pz,px,py)
        g.vertices = g.vertices[:, [2, 0, 1]]
        g.apply_translation([x0, 0, 0])
        slabs.append(g)
    body = union([leg_a, leg_b] + slabs)
    # glue-rivet holes: one revolved cutter each (plain cylinder, no
    # chamfer -> no tangency issue), margins past both faces (rule 5)
    cutters = []
    for x in HOLE_X:
        ca = trimesh.creation.cylinder(radius=HOLE_D / 2,
                                       height=LEG_T + 2 * OVERLAP,
                                       sections=48)
        ca.apply_translation([x, HOLE_POS, LEG_T / 2])
        cutters.append(ca)
        cb = trimesh.creation.cylinder(radius=HOLE_D / 2,
                                       height=LEG_T + 2 * OVERLAP,
                                       sections=48)
        cb.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
        cb.apply_translation([x, LEG_T / 2, HOLE_POS])
        cutters.append(cb)
    m = weld(difference(body, cutters))
    bbox = (LEG_LEN, LEG_DEPTH, LEG_DEPTH)
    return m, bbox, 8  # genus 8: 4 through-holes per leg


# ---------------- gates + export ----------------

def gate(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond


def export_and_gate(m, fname, bbox, genus):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, fname)
    m.export(path)
    r = trimesh.load(path, force="mesh")   # gate the RELOADED file
    ok = True
    ok &= gate("watertight", r.is_watertight)
    ok &= gate("single body", len(r.split(only_watertight=False)) == 1)
    want = 2 - 2 * genus
    ok &= gate("euler", r.euler_number == want,
               f"(got {r.euler_number}, want {want})")
    ext = r.bounding_box.extents
    ok &= gate("bbox", np.allclose(sorted(ext), sorted(bbox), atol=0.5),
               f"(got {np.round(ext, 2)}, want {bbox})")
    ok &= gate("fits AD3 bed", max(ext) <= 145.0, f"(max {max(ext):.1f})")
    vol = r.volume / 1000.0
    print(f"  volume {vol:.1f} cm^3  (~{vol * 1.04:.0f} g ABS / "
          f"{vol * 1.24:.0f} g PLA)")
    if not ok:
        print(f"GATES FAILED: {fname}")
        sys.exit(1)
    print(f"  -> {path}")
    return path


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else PART
    if part in ("ALL", "PLY_GAUGE"):
        print("PLY_GAUGE rev1")
        m, bbox, g = build_ply_gauge()
        export_and_gate(m, "PLY_GAUGE_rev1.stl", bbox, g)
    if part in ("ALL", "FLANGE_90"):
        print("FLANGE_90 rev1")
        m, bbox, g = build_flange_90()
        export_and_gate(m, "FLANGE90_rev1.stl", bbox, g)
