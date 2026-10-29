"""Tub-drain plug disc — Bathcraft fitting, Room Augmentations / tub-drain.

Single convex solid (no booleans). Built as a convex hull of vertex rings:
a chamfered cylinder is convex, so the hull IS the exact shape.
"""

import sys
import numpy as np
import trimesh

# =============== PARAMS ===============
PART = "plug47"
HOLE_ID = 47.50   # Brandon's caliper truth, 2026-08-29 — Bathcraft fitting inner ring ID
FIT_CLEAR = 0.30  # diametral slip-fit clearance, UNPROVEN on AD3 — this disc is its own gauge
OD = HOLE_ID - FIT_CLEAR          # 47.20 plug diameter at vertices
H = 5.00          # requested height
CHAMFER = 0.40    # bottom-edge 45° chamfer: kills elephant's foot, eases insertion
SEG = 256         # vertices ON the circle = max diameter exactly OD (correct for male ODs)
OUT = "stl/PLUG47_rev1.stl"
# ======================================


def ring(r, z, seg=SEG):
    a = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(seg, z)])


def build_plug():
    pts = np.vstack([
        ring(OD / 2 - CHAMFER, 0.0),   # bottom (bed) ring, pulled in by chamfer
        ring(OD / 2, CHAMFER),         # full diameter starts above the chamfer
        ring(OD / 2, H),               # top ring
    ])
    hull = trimesh.convex.convex_hull(pts)
    return hull


def weld(mesh):
    # conservative pre-export weld per boolean-pitfalls rule 8
    m = mesh.copy()
    m.vertices = np.round(m.vertices, 3)
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    if not m.is_watertight:
        return mesh
    return m


def ls_circle_radius(loop_xy):
    # least-squares circle fit (never vertex-centroid centering)
    x, y = loop_xy[:, 0], loop_xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(np.sqrt(c + cx**2 + cy**2))


def gates(path):
    m = trimesh.load(path)  # gate the RELOADED export, never the in-memory mesh
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok = ok and cond

    chk("watertight", m.is_watertight)
    chk("single body", m.body_count == 1, f"(bodies={m.body_count})")
    chk("euler=2 (genus 0, no holes)", m.euler_number == 2, f"(chi={m.euler_number})")
    chk("positive volume", m.volume > 0, f"(V={m.volume:.0f} mm^3)")

    ext = m.bounds[1] - m.bounds[0]
    chk("bbox XY = OD", np.allclose(ext[:2], OD, atol=0.02), f"(got {ext[0]:.3f} x {ext[1]:.3f})")
    chk("bbox Z = H", abs(ext[2] - H) < 0.02, f"(got {ext[2]:.3f})")
    chk("sits on Z=0", abs(m.bounds[0][2]) < 1e-3)

    # mid-height section: one loop, LS radius == OD/2
    sec = m.section(plane_origin=[0, 0, H / 2], plane_normal=[0, 0, 1])
    pts2d, _ = sec.to_2D()[0].vertices, None
    r_mid = ls_circle_radius(np.asarray(pts2d))
    chk("mid section r = OD/2", abs(r_mid - OD / 2) < 0.02, f"(r={r_mid:.3f}, want {OD/2:.3f})")
    chk("mid section 1 loop", len(sec.entities) == 1, f"(loops={len(sec.entities)})")

    # chamfer really present: near-bed section is smaller than OD
    sec0 = m.section(plane_origin=[0, 0, 0.05], plane_normal=[0, 0, 1])
    r0 = ls_circle_radius(np.asarray(sec0.to_2D()[0].vertices))
    want0 = OD / 2 - CHAMFER + 0.05  # 45 deg chamfer at z=0.05
    chk("bottom chamfer present", abs(r0 - want0) < 0.03, f"(r={r0:.3f}, want {want0:.3f})")

    # overhang audit: nothing steeper than 45 deg below horizontal,
    # excluding the bed-contact face (first layer is not an overhang)
    on_bed = m.triangles_center[:, 2] < 0.01
    down = (m.face_normals[:, 2] < -1e-6) & ~on_bed
    ang = np.degrees(np.arcsin(np.clip(-m.face_normals[down, 2], -1, 1)))
    worst = ang.max() if len(ang) else 0.0
    chk("support-free (no overhang >45deg)", worst <= 45.1, f"(steepest downward face {worst:.1f} deg below horizontal)")

    print(f"  volume {m.volume/1000:.2f} cm^3  (~{m.volume/1000*1.04:.1f} g ABS solid, less with infill)")
    return ok


if __name__ == "__main__":
    mesh = weld(build_plug())
    import os
    os.makedirs("stl", exist_ok=True)
    mesh.export(OUT)
    print(f"exported {OUT}: OD {OD:.2f} (hole {HOLE_ID:.2f} - clear {FIT_CLEAR:.2f}), H {H:.2f}, chamfer {CHAMFER:.2f}")
    print("gates on reloaded export:")
    sys.exit(0 if gates(OUT) else 1)
