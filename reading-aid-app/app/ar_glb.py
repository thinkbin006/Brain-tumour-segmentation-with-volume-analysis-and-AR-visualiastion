#!/usr/bin/env python3
"""Payload -> .glb converter for browser AR (companion to the patched app/server.py).

Reads the `mesh3d` section the N3 notebook already writes (no re-segmentation, no marching
cubes, no third-party packages -- stdlib only, like the rest of this app) and writes a binary
glTF (.glb) that <model-viewer> can place in AR on a phone.

Encoding, taken from the page's own decoder (MESHB / meshFix in the report HTML):
  * layer.v  base64 of uint16 LE triples, grid-index units = value / layer.scale
             (UNSIGNED: a signed read wraps coordinates past ~51.9 cells -- FIX v2.1)
  * layer.f  base64 of uint16 LE triples (uint32 when layer.f32), vertex indices
  * mesh3d.n is the mesh grid the indices live on; coordinates are crop-local, exactly
             like the .nii export (no world position/orientation is claimed)

Layer sets, matching the 3D panel's checkboxes:
  post (default)  brain + post_wt/post_tc/post_et   -- the delivered mask
  gt              brain + gt_wt/gt_tc/gt_et
  pred            brain + pred_wt/pred_tc/pred_et
  all             brain + all nine tumour layers (heavier; mainly for comparison)

Colours/alpha mirror the workstation (REGION_RGB in the report): brain glass, WT red,
TC amber, ET cyan -- except gt layers are green-tinted and pred layers orange-tinted so
the three sets stay distinguishable if you load `all`.

AR transform: plotly/grid space is Z-up-ish and unitless; glTF is Y-up and METRES.
We remap (x, y, z) -> (x, z, -y), centre on the brain, put the base on y=0 (the floor
plane AR anchors to), and scale the largest dimension to DISPLAY_M metres (20 cm -- a
natural brain size on a table). Without this the model would be ~96 metres wide.

    python3 app/ar_glb.py app/data/BraTS20_Training_001.json [--set post] [-o out.glb]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import sys
from array import array

DISPLAY_M = 0.20          # largest dimension of the model in AR, in metres

# (r, g, b, a) per layer -- same hues as the workstation 3D panel
FORMAT_VERSION = 3  # bump when GLB output changes; server cache includes this

STYLE = {
    "brain":   (0.67, 0.71, 0.78, 0.18),
    "post_wt": (0.90, 0.33, 0.29, 0.35),
    "post_tc": (0.82, 0.60, 0.13, 0.55),
    "post_et": (0.22, 0.77, 0.81, 0.85),
    "gt_wt":   (0.25, 0.73, 0.31, 0.35),
    "gt_tc":   (0.25, 0.73, 0.31, 0.55),
    "gt_et":   (0.25, 0.73, 0.31, 0.85),
    "pred_wt": (0.94, 0.53, 0.24, 0.35),
    "pred_tc": (0.94, 0.53, 0.24, 0.55),
    "pred_et": (0.94, 0.53, 0.24, 0.85),
}

SETS = {
    "post": ["brain", "post_wt", "post_tc", "post_et"],
    "gt":   ["brain", "gt_wt", "gt_tc", "gt_et"],
    "pred": ["brain", "pred_wt", "pred_tc", "pred_et"],
    "all":  ["brain", "post_wt", "post_tc", "post_et",
             "gt_wt", "gt_tc", "gt_et", "pred_wt", "pred_tc", "pred_et"],
}

# Canonical layer order (brain first so it renders as the outer shell).
LAYER_ORDER = ["brain",
               "gt_wt", "gt_tc", "gt_et",
               "pred_wt", "pred_tc", "pred_et",
               "post_wt", "post_tc", "post_et"]


def resolve_layers(payload: dict, layer_set: str = "post",
                   layers: str | list | None = None) -> list:
    """Which payload layers to export.

    `layers` (comma string or list) wins over `layer_set` and is exact: only
    whitelisted names that actually exist in this payload are kept, in
    canonical order, duplicates removed.  This is what lets the AR view mirror
    the workstation's layer checkboxes one-for-one, brain shell included or not.
    Raises ValueError if nothing usable is requested/found.
    """
    src = (payload.get("mesh3d") or {}).get("layers") or {}
    if layers:
        if isinstance(layers, str):
            layers = layers.split(",")
        asked = {str(x).strip() for x in layers if str(x).strip()}
        unknown = asked - set(LAYER_ORDER)
        if unknown:
            raise ValueError("unknown mesh3d layer(s): %s (valid: %s)"
                             % (", ".join(sorted(unknown)), ", ".join(LAYER_ORDER)))
        wanted = [k for k in LAYER_ORDER if k in asked and src.get(k)]
        missing = [k for k in LAYER_ORDER if k in asked and not src.get(k)]
        if missing:
            raise ValueError("payload has no mesh3d layer(s): %s" % ", ".join(missing))
    else:
        wanted = [k for k in SETS.get(layer_set, SETS["post"]) if src.get(k)]
    if not wanted:
        raise ValueError("payload has no usable mesh3d layers")
    return wanted


def _decode_layer(layer: dict):
    """(vertices flat [x,y,z,...] in grid units, faces flat index triples). Mirrors MESHB."""
    raw_v = base64.b64decode(layer["v"])
    u16 = array("H")
    u16.frombytes(raw_v[: len(raw_v) // 2 * 2])
    if sys.byteorder == "big":
        u16.byteswap()
    scale = float(layer.get("scale") or 1000.0)
    nv = int(layer["nv"])
    verts = [u16[i] / scale for i in range(nv * 3)]

    raw_f = base64.b64decode(layer["f"])
    faces = array("I" if layer.get("f32") else "H")
    faces.frombytes(raw_f[: len(raw_f) // faces.itemsize * faces.itemsize])
    if sys.byteorder == "big":
        faces.byteswap()
    return verts, list(faces[: int(layer["nf"]) * 3])


def _vertex_normals(verts, faces):
    """Smooth per-vertex normals (area-weighted), so the shells shade like the canvas panel."""
    n = array("f", bytes(len(verts) * 4))
    for t in range(0, len(faces), 3):
        a, b, c = faces[t] * 3, faces[t + 1] * 3, faces[t + 2] * 3
        ux, uy, uz = (verts[b] - verts[a], verts[b + 1] - verts[a + 1], verts[b + 2] - verts[a + 2])
        vx, vy, vz = (verts[c] - verts[a], verts[c + 1] - verts[a + 1], verts[c + 2] - verts[a + 2])
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        for i in (a, b, c):
            n[i] += nx; n[i + 1] += ny; n[i + 2] += nz
    for i in range(0, len(n), 3):
        l = math.sqrt(n[i] ** 2 + n[i + 1] ** 2 + n[i + 2] ** 2)
        if l > 0:
            n[i] /= l; n[i + 1] /= l; n[i + 2] /= l
        else:
            n[i], n[i + 1], n[i + 2] = 0.0, 1.0, 0.0   # degenerate patch: safe up-normal
    return n


def _frame(decoded: dict, wanted: list, display_m: float):
    """The exact transform build_glb applies: grid (a0,a1,a2) -> glTF metres.

        gltf_x = (a0 - centre[0]) * s
        gltf_y = (a2 - lo[2])     * s      (third grid axis becomes up; base on floor)
        gltf_z = -(a1 - centre[1]) * s

    Inverse (used by the AR page's tap-to-coordinates):
        a0 = gltf_x / s + centre[0]
        a1 = -gltf_z / s + centre[1]
        a2 = gltf_y / s + lo[2]
    """
    ref = decoded.get("brain", max(decoded.values(), key=lambda d: len(d[0])))[0]
    lo = [min(ref[i::3]) for i in range(3)]
    hi = [max(ref[i::3]) for i in range(3)]
    centre = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
    largest = max(hi[i] - lo[i] for i in range(3)) or 1.0
    return {"grid_lo": lo, "grid_hi": hi, "grid_centre": centre,
            "scale_m_per_grid_unit": display_m / largest,
            "frame_layer": "brain" if "brain" in decoded else max(
                decoded, key=lambda k: len(decoded[k][0]))}


def ar_metadata(payload: dict, layer_set: str = "post",
                layers: str | list | None = None,
                display_m: float = DISPLAY_M) -> dict:
    """Coordinate metadata for a payload + layer selection (no GLB built).

    Served at /api/AR/<pid>.json and embedded in the GLB extras, so the AR
    page can turn a tapped model point back into mesh-grid coordinates.
    `approx_full_volume_index` linearly rescales the coarse mesh grid onto
    vol3d.full -- an approximation for orientation, NOT a voxel-accurate or
    patient/RAS coordinate (the payload claims no world orientation).
    """
    src = (payload.get("mesh3d") or {}).get("layers") or {}
    wanted = resolve_layers(payload, layer_set, layers)
    decoded = {k: _decode_layer(src[k]) for k in wanted}
    frame = _frame(decoded, wanted, display_m)
    mesh3d = payload.get("mesh3d") or {}
    vol3d = payload.get("vol3d") or {}
    meta = payload.get("meta") or {}
    return {
        "layers": wanted,
        "layer_set": layer_set,
        "display_m": display_m,
        "mesh_n": mesh3d.get("n"),
        "mesh_full": vol3d.get("full"),
        "voxel_mm": (meta.get("export") or {}).get("voxel_mm"),
        "transform": frame,
        "coordinate_note": ("mesh3d grid indices (a0,a1,a2), crop-local; "
                            "approx_full_volume_index is a linear rescale onto vol3d.full "
                            "and is not a patient/RAS coordinate"),
    }


def build_glb(payload: dict, layer_set: str = "post",
              layers: str | list | None = None,
              display_m: float = DISPLAY_M) -> bytes:
    mesh3d = payload.get("mesh3d") or {}
    src_layers = mesh3d.get("layers") or {}
    wanted = resolve_layers(payload, layer_set, layers)

    decoded = {k: _decode_layer(src_layers[k]) for k in wanted}

    # --- global frame: centre + floor + metre scale (see _frame for the maths) -------
    _fr = _frame(decoded, wanted, display_m)
    lo, centre, s = _fr["grid_lo"], _fr["grid_centre"], _fr["scale_m_per_grid_unit"]

    # --- pack one binary buffer: per layer [positions | normals | indices] -----------
    bin_parts, views, accessors, meshes, nodes, materials = [], [], [], [], [], []
    offset = 0

    def pad4():
        nonlocal offset
        while offset % 4:
            bin_parts.append(b"\x00"); offset += 1

    for name in wanted:
        verts, faces = decoded[name]
        # Scene Viewer (Filament) is strict: reject/repair anything non-finite
        # or out of range here rather than handing the Google app a bad buffer.
        nv0 = len(verts) // 3
        if not faces or max(faces) >= nv0:
            raise ValueError("layer %s: face indices out of range" % name)
        if any(not math.isfinite(v) for v in verts):
            raise ValueError("layer %s: non-finite vertex" % name)
        # (x, y, z) grid -> (x, z, -y) glTF Y-up, centred, base on y=0, scaled to metres
        pos = array("f")
        for i in range(0, len(verts), 3):
            pos.append((verts[i] - centre[0]) * s)
            pos.append((verts[i + 2] - lo[2]) * s)       # grid z becomes up; base sits on floor
            pos.append(-(verts[i + 1] - centre[1]) * s)
        nor = _vertex_normals(verts, faces)
        nor_gltf = array("f")                            # same axis remap for normals
        for i in range(0, len(verts), 3):
            nor_gltf.append(nor[i]); nor_gltf.append(nor[i + 2]); nor_gltf.append(-nor[i + 1])
        use_u32 = max(faces) > 65535
        idx = array("I" if use_u32 else "H", faces)

        p_min = [min(pos[i::3]) for i in range(3)]
        p_max = [max(pos[i::3]) for i in range(3)]

        pad4(); v_pos = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": pos.itemsize * len(pos)})
        bin_parts.append(pos.tobytes()); offset += pos.itemsize * len(pos)

        pad4(); v_nor = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": nor_gltf.itemsize * len(nor_gltf)})
        bin_parts.append(nor_gltf.tobytes()); offset += nor_gltf.itemsize * len(nor_gltf)

        pad4(); v_idx = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": idx.itemsize * len(idx)})
        bin_parts.append(idx.tobytes()); offset += idx.itemsize * len(idx)

        nv, nf3 = len(pos) // 3, len(idx)
        a_pos = len(accessors)
        accessors.append({"bufferView": v_pos, "componentType": 5126, "count": nv,
                          "type": "VEC3", "min": p_min, "max": p_max})
        a_nor = len(accessors)
        accessors.append({"bufferView": v_nor, "componentType": 5126, "count": nv, "type": "VEC3"})
        a_idx = len(accessors)
        accessors.append({"bufferView": v_idx, "componentType": 5125 if use_u32 else 5123,
                          "count": nf3, "type": "SCALAR"})

        r, g, b, alpha = STYLE.get(name, (0.9, 0.4, 0.4, 0.8))
        # Scene Viewer (Google app) can crash on several nested BLEND layers.
        # Only the outer brain shell stays glass; every tumour layer is solid.
        if name != "brain":
            alpha = 1.0
        materials.append({
            "name": name,
            "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, alpha],
                                     "metallicFactor": 0.0, "roughnessFactor": 0.65},
            "alphaMode": "BLEND" if alpha < 1.0 else "OPAQUE",
            "doubleSided": True,
        })
        meshes.append({"name": name, "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nor},
            "indices": a_idx, "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})

    body = b"".join(bin_parts)
    gltf = {
        "asset": {"version": "2.0", "generator": "reading-aid ar_glb.py (mesh3d -> glb)"},
        "scene": 0,
        "scenes": [{"name": "segmentation", "nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes, "materials": materials,
        "buffers": [{"byteLength": len(body)}],
        "bufferViews": views, "accessors": accessors,
        "extras": {"layer_set": layer_set, "layers": wanted, "display_m": display_m,
                   "transform": _fr,
                   "mesh_n": mesh3d.get("n"),
                   "mesh_full": (payload.get("vol3d") or {}).get("full"),
                   "source": "mesh3d layers, crop-local grid units, no world orientation claimed"},
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(body)
    return b"".join([
        struct.pack("<III", 0x46546C67, 2, total),                    # magic 'glTF', version, length
        struct.pack("<II", len(js), 0x4E4F534A), js,                  # JSON chunk
        struct.pack("<II", len(body), 0x004E4942), body,              # BIN chunk
    ])


def convert_file(json_path: str, out_path: str, layer_set: str = "post",
                 layers: str | list | None = None) -> str:
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    data = build_glb(payload, layer_set, layers=layers)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(data)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Convert an N3 payload's mesh3d to an AR .glb")
    ap.add_argument("payload", help="report_<pid>.json (or <pid>.json in app/data)")
    ap.add_argument("--set", dest="layer_set", default="post",
                    choices=sorted(SETS), help="which layers (default: post = delivered mask)")
    ap.add_argument("--layers", help="exact layers, comma-separated, overrides --set "
                    "(e.g. brain,gt_wt,gt_tc,gt_et or just post_et; omit brain for no shell)")
    ap.add_argument("-o", "--out", help="output .glb (default: <pid>.<set>.glb next to the payload)")
    ap.add_argument("--size", type=float, default=DISPLAY_M, help="largest dimension in metres")
    a = ap.parse_args()
    tag = "layers" if a.layers else a.layer_set
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.payload)),
                                os.path.splitext(os.path.basename(a.payload))[0] + ".%s.glb" % tag)
    convert_file(a.payload, out, a.layer_set, layers=a.layers)
    print("wrote %s (%.2f MB, %s)" % (out, os.path.getsize(out) / 1e6,
                                      "layers=" + a.layers if a.layers else "set=" + a.layer_set))


if __name__ == "__main__":
    main()
