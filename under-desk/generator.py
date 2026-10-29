# Under-desk cord management - parametric part generator (Rev C).
# Usage: python generator.py            (builds PART below)
# Parts: key_gauge (more added as the project advances)
#
# Print: Flashforge Adventurer 3 (145mm usable), ABS, 0.2mm, no supports,
# orientation AS EXPORTED (Z up = build direction).
import numpy as np
import trimesh
import os

# =============== PARAMS ===============
PART = "leg_dock"

# LEG-DOCK: vertical power-strip dock on the 95x95 wood back leg.
# Strip body measured by Brandon (tape/caliper, 2026-08-16): ground truth.
STRIP_W = 42.5          # width across socket face
STRIP_T = 25.2          # thickness front-to-back
DOCK_CLEAR = 2.5        # pocket slack per side pair (strap snugs it)
PLATE_T = 5.0           # back plate thickness
WALL = 3.5              # pocket wall thickness
SHOE_PLATE_W = 56.0     # <= 95 leg face
SHOE_PLATE_H = 96.0
FLOOR_T = 6.0           # shoe floor slab
LIP_H = 18.0            # pocket wall height above floor
CORD_SLOT_W = 14.0      # open-front cord slot in floor
GUSSET_RUN = 20.0       # gusset extent below floor
SCREW_BORE = 4.5        # #8 wood screw clearance
SCREW_HEAD = 9.2        # countersink finish dia
CLIP_PLATE_W = 72.0     # wider than shoe: strap slots live outboard of horns
CLIP_PLATE_H = 40.0
HORN_H = 15.0           # top-clip side horn height
STRAP_SLOT = (4.5, 24.0)  # velcro slot (w, h)

# KEY-GAUGE: mushroom studs that test power-strip keyhole fit.
# (shank_dia, standoff) per stud; pips 1..6 identify them in order.
STUD_MATRIX = [
    (3.5, 2.5),   # 1
    (4.0, 2.5),   # 2
    (4.0, 3.5),   # 3
    (4.5, 2.5),   # 4
    (4.5, 3.5),   # 5
    (5.0, 3.0),   # 6
]
HEAD_EXTRA = 4.0        # head_dia = shank + this  (7.5..9.0 across matrix)
HEAD_T = 2.5            # head disc thickness (bed side, crisp)
COLLAR_R = 6.0          # fake-wall collar radius
COLLAR_T = 1.5          # collar cylindrical thickness above cone
COLLAR_FLAT = 0.8       # flat annulus width beyond shank before 45deg cone
FIN_BASE_W = 8.0        # fin width where it meets collar
FIN_TOP_W = 18.0        # fin width at top (45deg growth)
FIN_STRAIGHT = 8.0      # straight fin height above taper
FIN_T = 4.0             # fin thickness
PIP_D = 2.2
PIP_H = 0.6             # proud height (plus EMBED below surface)
EMBED = 0.3             # overlap for every union (boolean-pitfalls rule 3)
PITCH_X = 26.0          # stud spacing in a row
PITCH_Y = 30.0          # row spacing
SPRUE_W = 2.5           # snap sprue cross-section
SPRUE_H = 1.9
SEG = 128               # segments for revolved solids (mating surfaces)
OUT = os.path.join(os.path.dirname(__file__), "stl")
# ======================================

os.makedirs(OUT, exist_ok=True)


def revolve_profile(pts, seg=SEG):
    """Solid of revolution from an axis-touching (r,z) profile (r>=0,
    first and last points on axis). Single solid, no internal booleans."""
    pts = np.asarray(pts, dtype=float)
    assert pts[0][0] == 0 and pts[-1][0] == 0, "profile must touch axis"
    return trimesh.creation.revolve(pts[:, [0, 1]], sections=seg)


def stud(shank_d, standoff, n_pips):
    r_s = shank_d / 2.0
    r_h = (shank_d + HEAD_EXTRA) / 2.0
    z_head = HEAD_T
    z_gap = z_head + standoff                 # collar underside (flat zone)
    r_flat = r_s + COLLAR_FLAT
    z_cone = z_gap + (COLLAR_R - r_flat)      # 45 deg cone to collar radius
    z_collar = z_cone + COLLAR_T
    prof = [
        (0.0, 0.0),
        (r_h, 0.0),
        (r_h, z_head),
        (r_s, z_head),
        (r_s, z_gap),
        (r_flat, z_gap),
        (COLLAR_R, z_cone),
        (COLLAR_R, z_collar),
        (0.0, z_collar),
    ]
    body = revolve_profile(prof)

    # fin: extruded simple polygon (XZ), embedded EMBED into collar top
    taper = (FIN_TOP_W - FIN_BASE_W) / 2.0
    z0 = z_collar - EMBED
    fin2d = [
        (-FIN_BASE_W / 2, z0),
        (FIN_BASE_W / 2, z0),
        (FIN_TOP_W / 2, z0 + EMBED + taper),
        (FIN_TOP_W / 2, z0 + EMBED + taper + FIN_STRAIGHT),
        (-FIN_TOP_W / 2, z0 + EMBED + taper + FIN_STRAIGHT),
        (-FIN_TOP_W / 2, z0 + EMBED + taper),
    ]
    from shapely.geometry import Polygon
    fin = trimesh.creation.extrude_polygon(Polygon(fin2d), FIN_T)
    # extrude is XY->Z; rotate so polygon plane is XZ, thickness along Y
    fin.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, (1, 0, 0)))
    fin.apply_translation((0, FIN_T / 2, 0))

    z_fin_top = z0 + EMBED + taper + FIN_STRAIGHT
    # pips on the fin top edge (dice-style count, no glyph text)
    pips = []
    pitch = 2.7
    x0 = -(n_pips - 1) * pitch / 2.0
    for i in range(n_pips):
        p = trimesh.creation.cylinder(radius=PIP_D / 2, height=PIP_H + EMBED,
                                      sections=48)
        p.apply_translation((x0 + i * pitch, 0,
                             z_fin_top + (PIP_H + EMBED) / 2 - EMBED))
        pips.append(p)

    m = trimesh.boolean.union([body, fin] + pips, engine="manifold")
    return m, z_collar


def key_gauge():
    studs = []
    centers = []
    for i, (sd, so) in enumerate(STUD_MATRIX):
        m, _ = stud(sd, so, i + 1)
        row, col = divmod(i, 3)
        cx, cy = col * PITCH_X, row * PITCH_Y
        m.apply_translation((cx, cy, 0))
        studs.append(m)
        centers.append((cx, cy))

    # snap sprues at head level: tree topology (5 links, no loops -> genus 0)
    links = [(0, 1), (1, 2), (3, 4), (4, 5), (1, 4)]
    sprues = []
    for a, b in links:
        ax, ay = centers[a]
        bx, by = centers[b]
        ra = (STUD_MATRIX[a][0] + HEAD_EXTRA) / 2
        rb = (STUD_MATRIX[b][0] + HEAD_EXTRA) / 2
        v = np.array([bx - ax, by - ay], dtype=float)
        L = np.linalg.norm(v)
        u = v / L
        # from EMBED inside head A rim to EMBED inside head B rim
        s0 = np.array([ax, ay]) + u * (ra - EMBED)
        s1 = np.array([bx, by]) + u * (L - rb + EMBED - L) * -1  # placeholder
        s1 = np.array([bx, by]) - u * (rb - EMBED)
        mid = (s0 + s1) / 2
        length = np.linalg.norm(s1 - s0)
        box = trimesh.creation.box((length, SPRUE_W, SPRUE_H))
        ang = np.arctan2(u[1], u[0])
        box.apply_transform(trimesh.transformations.rotation_matrix(
            ang, (0, 0, 1)))
        box.apply_translation((mid[0], mid[1], SPRUE_H / 2 + 0.25))
        sprues.append(box)

    tree = trimesh.boolean.union(studs + sprues, engine="manifold")
    return tree


def export_and_gate(mesh, name, genus, loop_specs, custom_gates=None):
    path = os.path.join(OUT, name)
    # conservative weld (boolean-pitfalls rule 8)
    m = mesh.copy()
    m.vertices = np.round(m.vertices, 3)
    m.merge_vertices()
    keep = m.copy()
    if not m.is_watertight:
        m = mesh  # fall back to raw
    m.export(path)

    r = trimesh.load(path, force="mesh")     # gate the RELOADED file
    ok = True
    def gate(nm, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {nm} {detail}")
        ok = ok and cond
    gate("watertight", r.is_watertight)
    gate("winding", r.is_winding_consistent)
    gate("single body", len(r.split(only_watertight=False, repair=False)) == 1)
    gate("euler", r.euler_number == 2 - 2 * genus,
         f"(got {r.euler_number}, want {2 - 2 * genus})")
    src_ext = mesh.bounds[1] - mesh.bounds[0]
    dst_ext = r.bounds[1] - r.bounds[0]
    gate("bbox", np.allclose(src_ext, dst_ext, atol=0.5),
         f"(export {np.round(dst_ext,1)})")
    for z, n in loop_specs:
        sec = r.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        cnt = len(sec.discrete) if sec is not None else 0
        gate(f"section z={z}", cnt == n, f"(got {cnt}, want {n})")
    if custom_gates:
        for nm, cond, detail in custom_gates(r):
            gate(nm, cond, detail)
    if not ok:
        raise SystemExit(f"GATES FAILED for {name}")
    print(f"  -> {path}  ({r.volume/1000:.1f} cm^3, "
          f"~{r.volume/1000*1.04:.1f} g ABS)")
    return path


def key_gauge_custom(r):
    # inscribed shank diameter per stud at z just above tallest head
    z = HEAD_T + 1.0
    out = []
    for i, (sd, so) in enumerate(STUD_MATRIX):
        row, col = divmod(i, 3)
        cx, cy = col * PITCH_X, row * PITCH_Y
        sec = r.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        best = None
        for loop in sec.discrete:
            c = loop[:, :2].mean(axis=0)
            if np.linalg.norm(c - [cx, cy]) < 3:
                d = 2 * np.min(np.linalg.norm(loop[:, :2] - [cx, cy], axis=1))
                best = d
        okd = best is not None and abs(best - sd) < 0.15
        out.append((f"stud{i+1} shank Ø", okd,
                    f"(inscribed {best:.2f} vs {sd})" if best else "(missing)"))
    return out


def countersink_cutter(x, y):
    """Single revolved bore+countersink profile (boolean-pitfalls rule 2).
    Screw drives from the front face (z=PLATE_T) into the leg."""
    z_head = PLATE_T - 2.4
    prof = [
        (0.0, -1.0),
        (SCREW_BORE / 2, -1.0),
        (SCREW_BORE / 2, z_head),
        (SCREW_HEAD / 2, z_head + (SCREW_HEAD - SCREW_BORE) / 2),
        (SCREW_HEAD / 2, PLATE_T + 1.5),
        (0.0, PLATE_T + 1.5),
    ]
    c = revolve_profile(prof, seg=96)
    c.apply_translation((x, y, 0))
    return c


def box_at(w, h, d, cx, cy, cz):
    b = trimesh.creation.box((w, h, d))
    b.apply_translation((cx, cy, cz))
    return b


def leg_dock_shoe():
    """Bottom shoe. Print flat, plate on bed (back face = bed = flat on leg).
    Print frame: X width, Y = vertical-in-use, Z = out-from-leg."""
    W = SHOE_PLATE_W
    pw = STRIP_W + DOCK_CLEAR        # pocket interior width  (45.0)
    pd = STRIP_T + DOCK_CLEAR        # pocket interior depth  (27.7)
    z_out = PLATE_T + pd + WALL      # outer face of front lip (36.2)
    x0 = (W - pw) / 2 - WALL         # pocket outer left edge (4.0)
    x1 = W - x0
    yF0, yF1 = 30.0, 30.0 + FLOOR_T  # floor slab band
    plate = box_at(W, SHOE_PLATE_H, PLATE_T, W / 2, SHOE_PLATE_H / 2,
                   PLATE_T / 2)
    # floor slab (embed into plate)
    floor = box_at(x1 - x0, FLOOR_T, pd + WALL + EMBED, W / 2,
                   (yF0 + yF1) / 2, PLATE_T - EMBED + (pd + WALL + EMBED) / 2)
    # side walls
    walls = [
        box_at(WALL, FLOOR_T + LIP_H, pd + WALL + EMBED, x0 + WALL / 2,
               yF0 + (FLOOR_T + LIP_H) / 2, PLATE_T - EMBED + (pd + WALL + EMBED) / 2),
        box_at(WALL, FLOOR_T + LIP_H, pd + WALL + EMBED, x1 - WALL / 2,
               yF0 + (FLOOR_T + LIP_H) / 2, PLATE_T - EMBED + (pd + WALL + EMBED) / 2),
    ]
    # front lip (overlaps floor and both walls)
    lip = box_at(x1 - x0, FLOOR_T + LIP_H, WALL, W / 2,
                 yF0 + (FLOOR_T + LIP_H) / 2, z_out - WALL / 2)
    # gusset prism under floor: triangle in (Y,Z), thickness in X
    gp = [(yF0 - GUSSET_RUN, PLATE_T - EMBED), (yF0 + EMBED, PLATE_T - EMBED),
          (yF0 + EMBED, z_out - WALL)]
    from shapely.geometry import Polygon
    gus = trimesh.creation.extrude_polygon(Polygon(gp), 10.0)
    # extruded in XY->Z; remap: polygon plane (Y,Z), thickness X
    gus.apply_transform(np.array([[0, 0, 1, W / 2 - 5.0],
                                  [1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1]], dtype=float))
    body = trimesh.boolean.union([plate, floor] + walls + [lip, gus],
                                 engine="manifold")
    # cord slot: full-height front notch (floor + lip), open at the lip top
    # so it stays a notch, never a closed window (genus discipline)
    slot = box_at(CORD_SLOT_W, 27.0, (z_out - 16.0) + 1.0, W / 2,
                  41.5, 16.0 + ((z_out - 16.0) + 1.0) / 2)
    cutters = [slot, countersink_cutter(W / 2, 66.0),
               countersink_cutter(W / 2, 86.0)]
    return trimesh.boolean.difference([body] + cutters, engine="manifold")


def leg_dock_clip():
    """Top clip: plate + side horns + velcro slots. Print flat."""
    W = CLIP_PLATE_W
    pw = STRIP_W + DOCK_CLEAR
    x0 = (W - pw) / 2 - WALL
    x1 = W - x0
    plate = box_at(W, CLIP_PLATE_H, PLATE_T, W / 2, CLIP_PLATE_H / 2,
                   PLATE_T / 2)
    horns = [
        box_at(WALL, CLIP_PLATE_H, HORN_H + EMBED, x0 + WALL / 2,
               CLIP_PLATE_H / 2, PLATE_T - EMBED + (HORN_H + EMBED) / 2),
        box_at(WALL, CLIP_PLATE_H, HORN_H + EMBED, x1 - WALL / 2,
               CLIP_PLATE_H / 2, PLATE_T - EMBED + (HORN_H + EMBED) / 2),
    ]
    body = trimesh.boolean.union([plate] + horns, engine="manifold")
    sw, sh = STRAP_SLOT
    cutters = [
        box_at(sw, sh, PLATE_T + 2, x0 - 2.5 - sw / 2, CLIP_PLATE_H / 2,
               PLATE_T / 2),
        box_at(sw, sh, PLATE_T + 2, x1 + 2.5 + sw / 2, CLIP_PLATE_H / 2,
               PLATE_T / 2),
        countersink_cutter(W / 2, 12.0),
        countersink_cutter(W / 2, 28.0),
    ]
    return trimesh.boolean.difference([body] + cutters, engine="manifold")


def dock_pocket_gate(inner_w):
    def check(r):
        v = r.vertices
        mid = SHOE_PLATE_W / 2
        exp_l, exp_r = mid - inner_w / 2, mid + inner_w / 2
        band = v[(v[:, 1] > 25) & (v[:, 1] < 56) & (v[:, 2] > 4)]
        left = band[np.abs(band[:, 0] - exp_l) < 1.5]
        right = band[np.abs(band[:, 0] - exp_r) < 1.5]
        got = (right[:, 0].min() - left[:, 0].max()
               if len(left) and len(right) else 0)
        return [("pocket interior width", abs(got - inner_w) < 0.3,
                 f"(got {got:.2f}, want {inner_w})")]
    return check


if __name__ == "__main__":
    if PART == "leg_dock":
        shoe = leg_dock_shoe()
        # genus 2 (two screw bores; cord slot is open-ended = no genus)
        # z=2.5 mid-plate: outer loop + 2 bores = 3 loops
        export_and_gate(shoe, "LEGDOCK_SHOE_rev1.stl", genus=2,
                        loop_specs=[(2.5, 3)],
                        custom_gates=dock_pocket_gate(STRIP_W + DOCK_CLEAR))
        clip = leg_dock_clip()
        # genus 4 (2 bores + 2 strap slots); mid-plate: outer + 4 = 5 loops
        export_and_gate(clip, "LEGDOCK_CLIP_rev1.stl", genus=4,
                        loop_specs=[(2.5, 5)])
    elif PART == "key_gauge":
        m = key_gauge()
        # loops: z=1.2 cuts heads+sprues merged = 1 loop; z=HEAD_T+1.0 cuts
        # 6 shanks = 6 loops
        export_and_gate(m, "KEY_GAUGE_rev1.stl", genus=0,
                        loop_specs=[(1.2, 1), (HEAD_T + 1.0, 6)],
                        custom_gates=key_gauge_custom)
    else:
        raise SystemExit(f"unknown PART {PART}")
