#!/usr/bin/env python3
"""make_demo_case.py -- write a synthetic N3-shaped payload into app/data/.

Why this exists: the app must be verifiable BEFORE a real Kaggle run lands.  This builds a
phantom with the full story -- a main lesion, a MISSED satellite, a false-positive blob, an
unsure band around lambda* -- and writes every section the N3 contract defines (meta incl.
ai_review, slices, ortho, curves, vol3d, mesh3d, defaults, vol_curve, coverage, contact,
meta.export.rle), plus fixes/demo_expectations.json: voxel-level truths that the node test
(test_app_adapter.js) asserts the app reproduces through the real adapter.

    python3 code/make_demo_case.py            # writes app/data/synthetic-demo.json + index.json

Encodings used here (the app adapter also accepts the alternatives):
    vol3d layers      base64 of uint8, axis order (X,Y,Z), idx = ((i*ny)+j)*nx + k
    mesh3d pos        flat JSON list of floats in coarse-cell units; i/j/k face lists
    slices / ortho    lists of data:image/png URIs; axial row=y col=x, coronal row=z col=x,
                      sagittal row=z col=y; ortho stacks are 1:1 with the full axis, i0 offset
    export.rle        {shape:[X,Y,Z], runs:[value,length,...] flat pairs}
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "app", "data")
FIX = os.path.join(ROOT, "fixes")

X, Y, Z = 64, 64, 48            # full resolution, 1 mm voxels
NX, NY, NZ = 16, 16, 16         # coarse volume (visual aid).  CUBIC on purpose: the report
                                 # page's verified index formula idx=((a0*n1)+a1)*n0+a2 densely
                                 # packs an (X,Y,Z) cache exactly when the coarse grid is cubic
                                 # (real N3 payloads are); the app adapter ports that formula
                                 # byte-for-byte, so the demo must honour the same contract.
LAM = 0.20                      # lambda* the demo curves produce at alpha=0.10
THR_TC, THR_ET = 0.50, 0.60

xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")


def ell(c, r):
    return (((xx - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((zz - c[2]) / r[2]) ** 2)


def main():
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------- phantom
    q_brain = ell((32, 32, 24), (30, 30, 22))
    brain = q_brain <= 1.0
    q_main = ell((40, 28, 26), (8, 6.5, 6.5))
    q_sat = ell((18, 38, 20), (4, 4, 4))
    q_fp = ell((50, 48, 14), (3, 3, 3))

    gt_wt = (q_main <= 1.0) | (q_sat <= 1.0)
    gt_tc = q_main <= 0.55
    gt_et = q_main <= 0.30

    p = np.zeros((X, Y, Z))
    m = q_main <= 1.36
    p[m] = np.clip(0.95 - 0.55 * q_main[m], 0, 1)
    m = q_sat <= 1.3
    # satellite stays UNDER lambda*=0.20: the model does not find it (raw misses it too), it is
    # missed by the delivered mask, and its interior sits in the unsure band |p-lam|<=0.10
    p[m] = np.maximum(p[m], np.clip(0.185 - 0.12 * q_sat[m], 0, 1))
    m = q_fp <= 1.2
    p[m] = np.maximum(p[m], 0.70 - 0.25 * q_fp[m])                    # false positive: post includes it
    p = np.clip(p + rng.normal(0, 0.015, p.shape), 0, 1)
    p[~brain] *= 0.2

    post_wt = p >= 0.50                       # the frozen, delivered pipeline output
    post_tc = post_wt & (p >= 0.60) & (q_main <= 1.2)
    post_et = post_tc & (p >= 0.70) & (q_main <= 0.8)
    pred_wt = p >= LAM                        # AI outline at the operating point
    err = gt_wt ^ post_wt

    flair = np.where(brain, 110.0, 0.0)
    flair += np.where(q_main <= 1.6, 40.0, 0.0) + np.where(q_main <= 1.0, 30.0, 0.0)
    flair += np.where(gt_et, 60.0, 0.0) + np.where(q_sat <= 1.2, 26.0, 0.0)
    flair += np.where(q_fp <= 1.0, 45.0, 0.0)
    flair += rng.normal(0, 5.0, flair.shape)
    flair = np.clip(flair, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- png helpers
    def uri_grey(a):
        im = Image.fromarray(a.astype(np.uint8), "L")
        b = io.BytesIO(); im.save(b, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def uri_rgba(mask, rgb):
        rgba = np.zeros(mask.shape + (4,), np.uint8)
        rgba[mask] = (*rgb, 255)
        im = Image.fromarray(rgba, "RGBA")
        b = io.BytesIO(); im.save(b, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    slices = {"flair": [], "gt": [], "raw": [], "post": [], "err": []}
    for z in range(Z):                                        # axial: row=y, col=x
        slices["flair"].append(uri_grey(flair[:, :, z].T))
        slices["gt"].append(uri_rgba(gt_wt[:, :, z].T, (61, 220, 132)))
        slices["raw"].append(uri_rgba(pred_wt[:, :, z].T, (255, 169, 77)))
        slices["post"].append(uri_rgba(post_wt[:, :, z].T, (77, 171, 247)))
        slices["err"].append(uri_rgba(err[:, :, z].T, (255, 107, 107)))

    ortho = {"coronal": {"flair": [], "i0": 0}, "sagittal": {"flair": [], "i0": 0}}
    for y in range(Y):                                        # coronal: row=z, col=x
        ortho["coronal"]["flair"].append(uri_grey(flair[:, y, :].T))
    for x in range(X):                                        # sagittal: row=z, col=y
        ortho["sagittal"]["flair"].append(uri_grey(flair[x, :, :].T))

    # ---------------------------------------------------------------- coarse volumes
    def pool(a):
        return a.reshape(NX, X // NX, NY, Y // NY, NZ, Z // NZ).mean(axis=(1, 3, 5))
    # pooling factors: X 64->16 (4), Y 64->16 (4), Z 48->16 (3)

    c = {
        "wt": (pool(p) * 255).astype(np.uint8),
        "gtw": (pool(gt_wt.astype(np.float32)) * 255).astype(np.uint8),
        "gtc": (pool(gt_tc.astype(np.float32)) * 255).astype(np.uint8),
        "gte": (pool(gt_et.astype(np.float32)) * 255).astype(np.uint8),
        "tc": (pool(np.where(gt_tc | (q_main <= 1.2), p, 0.0)) * 255).astype(np.uint8),
        "et": (pool(np.where(gt_et | (q_main <= 0.8), p, 0.0)) * 255).astype(np.uint8),
        "postw": (pool(post_wt.astype(np.float32)) * 255).astype(np.uint8),
        "postc": (pool(post_tc.astype(np.float32)) * 255).astype(np.uint8),
        "poste": (pool(post_et.astype(np.float32)) * 255).astype(np.uint8),
    }
    vol3d = {"n": [NX, NY, NZ], "full": [X, Y, Z],
             "planes": {"axial": {"rows": 1, "cols": 0, "fix": 2},
                        "coronal": {"rows": 2, "cols": 0, "fix": 1},
                        "sagittal": {"rows": 2, "cols": 1, "fix": 0}}}
    for k, v in c.items():
        vol3d[k] = base64.b64encode(np.ascontiguousarray(v).tobytes()).decode()

    # ---------------------------------------------------------------- meshes (coarse cells)
    def layer(mask, smooth=0.8):
        f = ndi.gaussian_filter(pool(mask.astype(np.float32)), smooth)
        # small nested layers (TC/ET) lose their peak under pooling+smoothing: drop the iso
        # adaptively so they still get a surface instead of vanishing from the 3D view
        if f.max() >= 0.5:
            level = 0.5
        elif f.max() > 0.12:
            level = float(f.max() * 0.6)
        else:
            return None
        verts, faces, _, _ = marching_cubes(f, level=level)
        return {"pos": [round(float(t), 3) for t in verts.ravel()],
                "nv": int(len(verts)),
                "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist()}

    mesh_layers = {}
    for name, mask in [("brain", brain), ("gt_wt", gt_wt), ("gt_tc", gt_tc), ("gt_et", gt_et),
                       ("pred_wt", pred_wt), ("post_wt", post_wt)]:
        L = layer(mask)
        if L:
            mesh_layers[name] = L
    mesh3d = {"cell": [NX, NY, NZ], "layers": mesh_layers}

    # ---------------------------------------------------------------- numbers
    def stats(gt, ai):
        inter = int((gt & ai).sum())
        g, a = int(gt.sum()), int(ai.sum())
        dice = (2 * inter / (g + a)) if (g + a) else 0.0
        return {"agree": round(dice, 4), "covered": round(inter / g, 4) if g else None,
                "extra": round((a - inter) / a, 4) if a else None,
                "gt_mm3": g, "ai_mm3": a, "n_gt": g, "n_ai": a}

    labs, n_lab = ndi.label(gt_wt)
    sizes = sorted([int((labs == i).sum()) for i in range(1, n_lab + 1)], reverse=True)
    missed = int(sum(1 for i in range(1, n_lab + 1) if not (post_wt & (labs == i)).any()))
    et_frac = gt_et.sum() / max(1, gt_wt.sum())
    cnr = float((flair[gt_wt].mean() - flair[brain & ~gt_wt].mean()) / max(1e-6, flair[brain & ~gt_wt].std()))
    ai_review = (
        "DEMO PHANTOM. multifocal: %d separate reference lesions (the second is %d%% of the largest); "
        "the delivered outline misses %d of them; high-contrast stratum (CNR %.1f > cut 4.0); "
        "%s; reference WT %.2f cm3.  Rule-based impression from counted volumes and CNR -- not a "
        "histological or diagnostic claim." % (
            n_lab, round(100 * sizes[1] / sizes[0]) if len(sizes) > 1 else 0, missed, cnr,
            ("partial/ring-like enhancement inside a %d%% core sleeve" % round(100 * et_frac))
            if et_frac >= 0.12 else "almost non-enhancing",
            float(gt_wt.sum()) / 1000.0))

    thresholds = [round(0.05 + 0.05 * i, 2) for i in range(19)]
    fn_mean = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.17, 0.23, 0.30, 0.38,
               0.46, 0.55, 0.63, 0.71, 0.78, 0.85, 0.90, 0.94, 0.98]
    fp_mean = [0.62, 0.50, 0.40, 0.32, 0.25, 0.19, 0.145, 0.11, 0.08, 0.058,
               0.041, 0.028, 0.019, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001]
    # alpha=0.10 -> last i with (37/38)*fn+1/38 <= 0.10 is i=3 -> lambda* = 0.20

    runs, prev, ln = [], None, 0
    for v in post_wt.reshape(-1).astype(np.uint8):
        if v == prev:
            ln += 1
        else:
            if prev is not None:
                runs += [int(prev), int(ln)]
            prev, ln = v, 1
    runs += [int(prev), int(ln)]

    montage_zs = [22, 24, 26, 28, 18, 20]
    tiles = [flair[:, :, z].T for z in montage_zs]
    mont = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    contact = {"png": uri_grey(mont), "caption": "demo contact sheet"}

    payload = {
        "meta": {
            "case_id": "synthetic-demo", "lam": LAM, "thr_tc": THR_TC, "thr_et": THR_ET,
            "z_i0": 0, "gt_volume": int(gt_wt.sum()), "post_volume": int(post_wt.sum()),
            "ai": {"WT": stats(gt_wt, post_wt), "TC": stats(gt_tc, post_tc), "ET": stats(gt_et, post_et)},
            "cert_scope": "demo phantom - no certificate", "n_cal": 37,
            "ai_review": ai_review,
            "cnr": round(cnr, 2), "n_pred_components": int(ndi.label(pred_wt)[1]),
            "export": {"rle": {"shape": [X, Y, Z], "runs": runs}},
        },
        "defaults": {"alpha": 0.10, "beta": 0.20, "lam": LAM, "win": 1, "lvl": 1},
        "curves": {"n_cal": 37, "thresholds": thresholds, "fn_mean": fn_mean, "fp_mean": fp_mean},
        "slices": slices,
        "ortho": ortho,
        "vol3d": vol3d,
        "mesh3d": mesh3d,
        "vol_curve": {"lam": thresholds,
                      "wt_cm3": [round(float((p >= t).sum()) / 1000, 3) for t in thresholds],
                      "post_cm3": [round(float(post_wt.sum()) / 1000, 3)] * len(thresholds)},
        "coverage": {"WT": stats(gt_wt, post_wt), "note": "demo coverage block"},
        "contact": contact,
    }

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "synthetic-demo.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    idx_path = os.path.join(DATA, "index.json")
    man = {}
    if os.path.exists(idx_path):
        try:
            man = json.load(open(idx_path, encoding="utf-8"))
        except Exception:
            man = {}
    man["synthetic-demo"] = {"tag": "demo phantom", "patient": "SYNTH-DEMO",
                             "why": "missed satellite + false positive + unsure band; exercises every payload section"}
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)

    # ---------------------------------------------------------------- expectations for the node test
    def axial_css(cxyz, el=100):
        cc, rr, fi = cxyz
        # slider i is the FULL z (48 axial slices, z_i0=0); fi = round(i*NZ/Z) -> i = fi*(Z//NZ)
        return {"i": fi * (Z // NZ), "x": (cc + 0.5) / NX * el, "y": (rr + 0.5) / NY * el}

    def flags(i, j, k):
        gw = c["gtw"][i, j, k] > 127; pw = c["wt"][i, j, k] / 255 >= LAM
        de = (c["postw"][i, j, k] > 127) or (c["postc"][i, j, k] > 127) or (c["poste"][i, j, k] > 127)
        pp = c["wt"][i, j, k] / 255
        return {"gw": bool(gw), "pw": bool(pw), "del": bool(de), "p": round(float(pp), 4),
                "unc": bool(abs(pp - LAM) <= max(0.10, 0.35 * LAM))}

    def find(pred):
        hits = np.argwhere(pred(c))
        return tuple(int(t) for t in hits[len(hits) // 2]) if len(hits) else None

    v_agree = find(lambda a: (a["gtw"] > 127) & (a["postw"] > 127) & (a["wt"] > 200))
    v_miss = find(lambda a: (a["gtw"] > 127) & (a["postw"] <= 127) & (np.abs(a["wt"] / 255 - LAM) <= 0.10))
    v_extra = find(lambda a: (a["gtw"] <= 127) & (a["postw"] > 127))
    v_bg = find(lambda a: (a["wt"] < 25) & (a["gtw"] <= 127) & (a["postw"] <= 127))
    probes = []
    for name, v in [("agree", v_agree), ("missed-satellite", v_miss), ("extra-fp", v_extra), ("background", v_bg)]:
        assert v is not None, "demo phantom lost the %s voxel" % name
        css = axial_css(v)
        probes.append({"name": name, "plane": "axial", "voxel": list(v), "css": css,
                       "expect": flags(*v)})
    # the same agree voxel through the coronal axis map: rows=a2(Z), cols=a0(X), fix=a1(Y)
    i0, j0, k0 = v_agree
    # coronal: rows=a2 (Z coarse), cols=a0 (X coarse), fix=a1 (Y); slider over 64 full-Y images,
    # fi = round(i*NY/Y) -> i = j0*(Y//NY); css y maps rows=Z coarse, x maps cols=X coarse
    probes.append({"name": "agree-via-coronal", "plane": "coronal", "voxel": [i0, j0, k0],
                   "css": {"i": j0 * (Y // NY), "x": (i0 + 0.5) / NX * 100, "y": (k0 + 0.5) / NZ * 100},
                   "expect": flags(i0, j0, k0)})

    exp = {
        "lam_check": {"alpha": 0.10, "lam": LAM},
        "probes": probes,
        "head_check_bad": [],
        "post_full_count": int(post_wt.sum()),
        "rle_shape": [X, Y, Z],
        "sections_consumed": ["meta", "defaults", "curves", "slices", "ortho", "vol3d", "mesh3d",
                              "vol_curve", "coverage", "contact"],
    }
    os.makedirs(FIX, exist_ok=True)
    with open(os.path.join(FIX, "demo_expectations.json"), "w", encoding="utf-8") as fh:
        json.dump(exp, fh, indent=1)

    print("  wrote %s (%.1f KB)" % (out, os.path.getsize(out) / 1024))
    print("  lesions: %d reference components, %d missed by post, CNR %.1f, ET/WT %.0f%%"
          % (n_lab, missed, cnr, 100 * et_frac))
    print("  lambda* at alpha=0.10: %.2f (thresholds[3])" % LAM)
    print("  probe voxels: " + ", ".join("%s@%s" % (p["name"], p["voxel"]) for p in probes))
    print("  wrote %s" % os.path.join(FIX, "demo_expectations.json"))


if __name__ == "__main__":
    main()
