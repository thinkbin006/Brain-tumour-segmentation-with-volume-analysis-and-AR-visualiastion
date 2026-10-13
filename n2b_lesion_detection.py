# %% [markdown] N2b - lesion detection by SIZE, and what a component-size filter would cost (CPU, no GPU)
#
# Why: N2's `missed_lesions` counts every connected component of the reference whole-tumour mask, including
# specks of a few voxels, so "56 % of lesions missed" cannot tell a missed 30 cm3 tumour from a missed
# 20-voxel annotation island.  This breaks detection down by lesion volume, by contrast stratum, and by how
# much tumour volume the missed components actually carry.  It also measures the false components and what a
# component-size filter would remove / delete, and (on the calibration split) picks the certified filter size.
#
# Inputs = the same attachments N3 needs: the N0 cache (`*_seg.npy`), the N2 probabilities (`<pid>.npz`),
# and N2's outputs (certificates.json, frozen_postproc.json, dataset_split.json, strata.csv, test_csvs/).
# It re-creates N2's delivered masks with N2's own rules and then CHECKS ITSELF against `<arm>_test.csv`
# (n_gt, missed_lesions, false_components must match scan by scan).  If that check fails, do not trust the rest.

# %% SETTINGS ----------------------------------------------------------------
ARM          = "base"            # arm to analyse ("base" is the one I would deliver; also try "ens")
SPLITS       = ("val", "test")   # val = calibration of the filter, test = evaluation
MEANINGFUL_CM3 = 1.0             # a judgement call, not a standard: lesions >= this are "clinically meaningful"
BINS_CM3     = (0.1, 1.0, 5.0, 20.0)
FILTER_ALPHAS = (0.01, 0.05, 0.10)   # collateral-damage budgets (fraction of true tumour volume the filter may delete)
DELTA_GRID   = (1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000)   # voxels (1 mm^3 each)
WORK         = "/kaggle/working"
INPUT_ROOT   = "/kaggle/input"
OUT          = WORK + "/lesion_report"
CACHE_BASE   = None   # None = auto-discover (dirs named cache_base* holding *_mri.npy), or give a list of dirs
PROBS_DIR    = None   # None = auto-discover (dir named probs* holding <arm>/<pid>.npz), or give the dir
VOXEL_MM3    = 1.0    # BraTS is 1 mm isotropic

# %% CODE ---------------------------------------------------------------------
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import ndimage

CONN26 = np.ones((3, 3, 3), bool)
WT_LABELS = (1, 2, 4)


# ---- the pieces of safepp that define N2's delivered mask (copied verbatim in logic) -----------------
def region_thresholds(lam, params=None):
    p = params or {}
    out = {"WT": float(lam)}
    for key, cap in (("thr_tc", "TC"), ("thr_et", "ET")):
        v = p.get(key)
        out[cap] = float(lam) if v is None else float(v)
    return out


def labels_from_probs(p_wt, p_tc, p_et, thr):
    t_wt = float(thr["WT"]); t_tc = float(thr.get("TC", t_wt)); t_et = float(thr.get("ET", t_tc))
    wt, tc, et = p_wt >= t_wt, p_tc >= t_tc, p_et >= t_et
    seg = np.zeros(p_wt.shape, np.uint8)
    seg[wt] = 2; seg[wt & tc] = 1; seg[wt & tc & et] = 4
    return seg


def components(mask):
    lab, n = ndimage.label(np.asarray(mask, bool), structure=CONN26)
    return lab.astype(np.int32), int(n)


def filter_components(mask, min_voxels):
    mask = np.asarray(mask, bool)
    if min_voxels <= 1:
        return mask.copy()
    lab, n = components(mask)
    if n == 0:
        return mask.copy()
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_voxels
    keep[0] = False
    return keep[lab]


def postproc_seg(seg, params):
    REG = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}
    seg = np.asarray(seg).copy()
    mv = (params or {}).get("min_vox") or {}
    wt_keep = filter_components(np.isin(seg, REG["WT"]), int(mv.get("WT", 1)))
    tc_keep = filter_components(np.isin(seg, REG["TC"]), int(mv.get("TC", 1)))
    et_keep = filter_components(np.isin(seg, REG["ET"]), int(mv.get("ET", 1)))
    if (params or {}).get("keep_largest_wt"):
        lab, n = components(wt_keep)
        if n > 1:
            wt_keep = lab == int(np.argmax(np.bincount(lab.ravel(), minlength=n + 1)[1:]) + 1)
    out = np.zeros_like(seg)
    out[wt_keep] = seg[wt_keep]
    out[~wt_keep] = 0
    out[np.isin(seg, REG["ET"]) & ~et_keep & wt_keep] = 1
    out[(np.isin(seg, REG["TC"]) & ~tc_keep) & (out != 0)] = 2
    m = (params or {}).get("min_et_voxels", 0)
    if m and m > 0 and int((out == 4).sum()) < m:
        out[out == 4] = 1
    return out


def certified_filter_threshold(R, alpha, thresholds, B=1.0):
    """Largest delta whose calibration bound n/(n+1)*mean + B/(n+1) stays <= alpha (CRC, same as safepp)."""
    R = np.asarray(R, float); n = R.shape[0]
    mean = R.mean(axis=0); bound = (n / (n + 1.0)) * mean + B / (n + 1.0)
    ok = np.where(bound <= alpha)[0]
    j = int(ok[-1]) if len(ok) else 0
    return dict(delta=float(thresholds[j]), feasible=bool(len(ok)), bound=float(bound[j]), mean=float(mean[j]),
                floor=B / (n + 1.0), n_cal=n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


# ---- locating inputs (same logic as N3) --------------------------------------------------------------
def _find_dirs(prefix, probe):
    hits = [p for p in glob.glob(os.path.join(INPUT_ROOT, "**", "*"), recursive=True)
            if os.path.isdir(p) and os.path.basename(p).startswith(prefix) and glob.glob(os.path.join(p, probe))]
    return sorted(set(hits))


def _first(name, extra=()):
    hits = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", name), recursive=True)) + list(extra)
    for h in hits:
        if os.path.exists(h):
            return h
    return None


def locate():
    cache = list(CACHE_BASE) if isinstance(CACHE_BASE, (list, tuple)) else (
        [CACHE_BASE] if CACHE_BASE else (_find_dirs("cache_base", "*_seg.npy") or [os.path.join(WORK, "cache_base")]))
    probs = PROBS_DIR or next(iter(_find_dirs("probs", os.path.join("*", "*.npz"))), None) or os.path.join(WORK, "probs")
    armdir = os.path.join(probs, ARM)
    if not os.path.isdir(armdir):
        c = sorted(glob.glob(os.path.join(probs, "*%s*" % ARM)))
        armdir = c[0] if c else armdir
    assert os.path.isdir(armdir), "no probability folder for arm %r under %s" % (ARM, probs)
    f = lambda n, *x: _first(n, [os.path.join(WORK, n)] + list(x))
    cert = f("certificates.json"); post = f("frozen_postproc.json"); split = f("dataset_split.json")
    assert cert and post and split, "certificates.json / frozen_postproc.json / dataset_split.json not found"
    csv = _first("%s_test.csv" % ARM, [os.path.join(WORK, "test_csvs", "%s_test.csv" % ARM)])
    strata = _first("strata.csv", [os.path.join(WORK, "strata.csv")])
    return dict(cache=cache, armdir=armdir, cert=cert, post=post, split=split, csv=csv, strata=strata)


def seg_path(cache, pid):
    for r in cache:
        p = os.path.join(r, "%s_seg.npy" % pid)
        if os.path.exists(p):
            return p
    return None


# ---- one case ------------------------------------------------------------------------------------------
def analyse_case(pid, lam, P, POST):
    z = np.load(os.path.join(P["armdir"], "%s.npz" % pid))
    seg = np.load(seg_path(P["cache"], pid))
    raw = labels_from_probs(z["p_wt"], z["p_tc"], z["p_et"], region_thresholds(lam, POST))
    pred = postproc_seg(raw, POST) > 0                      # delivered whole-tumour mask
    gt = seg > 0
    lg, ng = components(gt); lp, npc = components(pred)
    sg = np.bincount(lg.ravel(), minlength=ng + 1)[1:]; sp = np.bincount(lp.ravel(), minlength=npc + 1)[1:]
    ov_g = np.bincount(lg[pred], minlength=ng + 1)[1:] if ng else np.zeros(0, int)      # predicted voxels inside each GT comp
    ov_p = np.bincount(lp[gt], minlength=npc + 1)[1:] if npc else np.zeros(0, int)      # GT voxels inside each pred comp
    ov = gt & pred
    pairs = {}
    if ov.any():
        key = lg[ov].astype(np.int64) * (npc + 1) + lp[ov]
        u, c = np.unique(key, return_counts=True)
        for k, v in zip(u, c):
            pairs.setdefault(int(k // (npc + 1)), []).append((int(k % (npc + 1)), int(v)))
    gt_rows = [dict(pid=pid, comp=i + 1, size_vox=int(sg[i]), size_cm3=float(sg[i]) * VOXEL_MM3 / 1000.0,
                    detected=bool(ov_g[i] > 0), recall=float(ov_g[i]) / float(sg[i]),
                    found_well=bool(ov_g[i] >= 0.5 * sg[i])) for i in range(ng)]
    fp_rows = [dict(pid=pid, comp=j + 1, size_vox=int(sp[j]), size_cm3=float(sp[j]) * VOXEL_MM3 / 1000.0,
                    false=bool(ov_p[j] == 0), gt_overlap_vox=int(ov_p[j])) for j in range(npc)]
    # filter table: for each delta, what a "remove predicted components < delta" step would do to this case
    gvol = float(gt.sum())
    filt = []
    for d in DELTA_GRID:
        removed = [j for j in range(npc) if sp[j] < d]
        gt_deleted = float(sum(ov_p[j] for j in removed))
        false_removed = sum(1 for j in removed if ov_p[j] == 0)
        lost = 0
        for i in range(ng):
            if ov_g[i] > 0 and not any(sp[pj - 1] >= d for pj, _ in pairs.get(i + 1, [])):
                lost += 1                                    # was detected, would lose every predicted voxel
        filt.append(dict(pid=pid, delta=d, gt_deleted_frac=(gt_deleted / gvol) if gvol else 0.0,
                         false_removed=false_removed, false_total=int(sum(ov_p == 0)), lesions_lost=lost,
                         lesions_lost_meaningful=sum(1 for i in range(ng) if ov_g[i] > 0
                                                     and not any(sp[pj - 1] >= d for pj, _ in pairs.get(i + 1, []))
                                                     and sg[i] * VOXEL_MM3 / 1000.0 >= MEANINGFUL_CM3)))
    return gt_rows, fp_rows, filt, dict(pid=pid, n_gt=ng, n_pred=npc, missed=int((ov_g == 0).sum()),
                                        false=int((ov_p == 0).sum()), gt_vol=gvol)


# ---- main ----------------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    P = locate()
    CERT = json.load(open(P["cert"])); POST = json.load(open(P["post"])); SPLIT = json.load(open(P["split"]))
    CA = CERT[ARM]; LAM = float(CA["lam"]); CUT = (CA.get("strata") or {}).get("cut")
    strata = pd.read_csv(P["strata"]).set_index("pid") if P["strata"] else None
    csv = pd.read_csv(P["csv"]).set_index("pid") if P["csv"] else None
    print("arm %s | pooled lambda* %.2f | frozen post-processing %s | CNR cut %s" % (ARM, LAM, POST, CUT))
    G, F, FT, S = {}, {}, {}, {}
    for split in SPLITS:
        ids = [p for p in (SPLIT.get(split) or []) if os.path.exists(os.path.join(P["armdir"], "%s.npz" % p))
               and seg_path(P["cache"], p)]
        missing = len(SPLIT.get(split) or []) - len(ids)
        print("split %-4s: %d scans%s" % (split, len(ids), (" (%d without probabilities/cache -> skipped)" % missing) if missing else ""))
        gr, fr, ft, sr = [], [], [], []
        for k, pid in enumerate(ids):
            lam = float(csv.loc[pid, "lam"]) if (split == "test" and csv is not None and pid in csv.index) else LAM
            a, b, c, d = analyse_case(pid, lam, P, POST)
            cnr = float(strata.loc[pid, "wt_cnr"]) if (strata is not None and pid in strata.index) else float("nan")
            for r in a: r.update(cnr=cnr, low=bool(CUT is not None and np.isfinite(cnr) and cnr <= CUT))
            gr += a; fr += b; ft += c; sr.append(d)
            if (k + 1) % 10 == 0: print("   ...%d/%d" % (k + 1, len(ids)))
        G[split], F[split], FT[split], S[split] = pd.DataFrame(gr), pd.DataFrame(fr), pd.DataFrame(ft), pd.DataFrame(sr)
        G[split].to_csv(os.path.join(OUT, "gt_components_%s_%s.csv" % (ARM, split)), index=False)
        F[split].to_csv(os.path.join(OUT, "pred_components_%s_%s.csv" % (ARM, split)), index=False)

    # ---- self-check against N2's own numbers --------------------------------------------------------
    ok = None
    if csv is not None and "test" in S:
        m = S["test"].set_index("pid").join(csv[["n_gt", "missed_lesions", "false_components"]], rsuffix="_n2", how="inner")
        bad = m[(m.n_gt != m.n_gt_n2) | (m.missed != m.missed_lesions) | (m.false != m.false_components)]
        ok = (len(bad) == 0 and len(m) > 0)
        print("\nSELF-CHECK vs N2 %s_test.csv: %d/%d scans reproduce n_gt, missed_lesions and false_components exactly -> %s"
              % (ARM, len(m) - len(bad), len(m), "OK" if ok else "MISMATCH: the analysis below does NOT describe N2's masks"))
        if len(bad): print(bad[["n_gt", "n_gt_n2", "missed", "missed_lesions", "false", "false_components"]].head(8).to_string())
    else:
        print("\n(no test CSV found: could not self-check against N2)")

    # ---- detection by size ---------------------------------------------------------------------------
    edges = [0.0] + list(BINS_CM3) + [np.inf]
    labels = ["< %g cm3" % edges[1]] + ["%g-%g cm3" % (edges[i], edges[i + 1]) for i in range(1, len(edges) - 2)] + [">= %g cm3" % edges[-2]]
    out = {"arm": ARM, "self_check_ok": ok, "meaningful_cm3": MEANINGFUL_CM3}
    for split in ("test",):
        if split not in G: continue
        g = G[split].copy(); g["bin"] = pd.cut(g.size_cm3, edges, labels=labels, right=False)
        print("\n=== %s split, arm %s: detection of reference lesions by volume (detected = at least 1 predicted voxel inside)" % (split, ARM))
        print("%-14s %5s %9s %20s %11s %14s" % ("lesion size", "n", "detected", "95% CI", "found well", "vol. share"))
        tot_vol = g.size_vox.sum(); rows = []
        for lab in labels:
            s = g[g.bin == lab]
            if not len(s): continue
            k = int(s.detected.sum()); lo, hi = wilson(k, len(s))
            rows.append(dict(bin=lab, n=len(s), detected=k, rate=k / len(s), ci_lo=lo, ci_hi=hi,
                             found_well=float(s.found_well.mean()), vol_share=float(s.size_vox.sum() / tot_vol)))
            print("%-14s %5d %8.0f%% %9.0f%% - %3.0f%% %10.0f%% %13.1f%%" % (lab, len(s), 100 * k / len(s), 100 * lo, 100 * hi,
                  100 * s.found_well.mean(), 100 * s.size_vox.sum() / tot_vol))
        out["detection_by_size"] = rows
        mm = g[g.size_cm3 >= MEANINGFUL_CM3]
        out["meaningful"] = dict(n=len(mm), detected=int(mm.detected.sum()), rate=float(mm.detected.mean()) if len(mm) else None,
                                 ci=wilson(int(mm.detected.sum()), len(mm)), missed_volume_share=float(mm[~mm.detected].size_vox.sum() / tot_vol))
        print("\nlesions >= %g cm3: %d, detected %d (%.0f%%, 95%% CI %.0f-%.0f%%); missed ones carry %.1f%% of all reference tumour volume"
              % (MEANINGFUL_CM3, len(mm), mm.detected.sum(), 100 * mm.detected.mean() if len(mm) else float("nan"),
                 100 * out["meaningful"]["ci"][0], 100 * out["meaningful"]["ci"][1], 100 * out["meaningful"]["missed_volume_share"]))
        pidm = mm[~mm.detected].groupby("pid").size()
        out["scans_missing_a_meaningful_lesion"] = int(len(pidm)); print("scans missing at least one such lesion: %d / %d" % (len(pidm), g.pid.nunique()))
        print("\nby contrast stratum (lesions >= %g cm3):" % MEANINGFUL_CM3)
        for name, flag in (("low-contrast", True), ("high-contrast", False)):
            s = mm[mm.low == flag]
            if len(s):
                k = int(s.detected.sum()); lo, hi = wilson(k, len(s))
                print("  %-14s n=%3d detected %3d = %3.0f%% [%.0f-%.0f%%]" % (name, len(s), k, 100 * k / len(s), 100 * lo, 100 * hi))
                out.setdefault("by_stratum", {})[name] = dict(n=len(s), detected=k, rate=k / len(s), ci=[lo, hi])
        f = F[split]; ff = f[f["false"]]
        print("\nfalse components (predicted, no reference overlap): %d in %d scans; size >= %g cm3: %d; median size %.0f voxels"
              % (len(ff), ff.pid.nunique(), MEANINGFUL_CM3, int((ff.size_cm3 >= MEANINGFUL_CM3).sum()), ff.size_vox.median() if len(ff) else 0))
        out["false_components"] = dict(n=len(ff), scans=int(ff.pid.nunique()), n_meaningful=int((ff.size_cm3 >= MEANINGFUL_CM3).sum()))

    # ---- component-size filter: certify on val, evaluate on test --------------------------------------
    if "val" in FT and "test" in FT and len(S["val"]) >= 5:
        pv = FT["val"].pivot(index="pid", columns="delta", values="gt_deleted_frac")[list(DELTA_GRID)]
        print("\n=== component-size filter (remove predicted components smaller than delta voxels)")
        print("calibration on %d val scans; the finite-sample term alone costs %.3f of any alpha, so alpha below that is unreachable"
              % (len(pv), 1.0 / (len(pv) + 1)))
        ft_t = FT["test"]; res = []
        print("%-7s %-9s %-10s %-13s %-14s %-17s %s" % ("alpha", "delta", "feasible", "val bound", "test: tumour", "test: false comps", "test: lesions >=%gcm3 lost" % MEANINGFUL_CM3))
        print("%-7s %-9s %-10s %-13s %-14s %-17s" % ("", "(voxels)", "", "", "volume deleted", "removed"))
        for a in FILTER_ALPHAS:
            c = certified_filter_threshold(pv.values, a, np.array(DELTA_GRID, float)); d = int(c["delta"])
            t = ft_t[ft_t.delta == d]
            row = dict(alpha=a, delta=d, feasible=c["feasible"], val_bound=c["bound"], test_deleted=float(t.gt_deleted_frac.mean()),
                       test_false_removed=int(t.false_removed.sum()), test_false_total=int(t.false_total.sum()),
                       test_meaningful_lost=int(t.lesions_lost_meaningful.sum()))
            res.append(row)
            print("%-7.2f %-9d %-10s %-13.3f %-14s %-17s %d" % (a, d, "yes" if c["feasible"] else "NO (floor)", c["bound"],
                  "%.2f%%" % (100 * row["test_deleted"]), "%d of %d" % (row["test_false_removed"], row["test_false_total"]), row["test_meaningful_lost"]))
        out["filter"] = res
        print("\nfull test-split trade-off table (descriptive, NOT certified):")
        tt = ft_t.groupby("delta").agg(deleted=("gt_deleted_frac", "mean"), false_removed=("false_removed", "sum"),
                                       false_total=("false_total", "sum"), lost=("lesions_lost", "sum"), lost_m=("lesions_lost_meaningful", "sum"))
        tt["deleted"] = (100 * tt.deleted).round(2); print(tt.to_string())
    else:
        print("\n(filter analysis skipped: needs val AND test probabilities)")

    json.dump(out, open(os.path.join(OUT, "lesion_summary_%s.json" % ARM), "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        r = out.get("detection_by_size") or []
        if r:
            x = np.arange(len(r)); rate = np.array([q["rate"] for q in r]); lo = np.array([q["ci_lo"] for q in r]); hi = np.array([q["ci_hi"] for q in r])
            ax[0].bar(x, 100 * rate, color="#58a6ff"); ax[0].errorbar(x, 100 * rate, yerr=[100 * (rate - lo), 100 * (hi - rate)], fmt="none", ecolor="k", capsize=3)
            ax[0].set_xticks(x); ax[0].set_xticklabels(["%s\nn=%d" % (q["bin"], q["n"]) for q in r], fontsize=8)
            ax[0].set_ylabel("reference lesions detected (%)"); ax[0].set_ylim(0, 105); ax[0].set_title("detection by lesion volume (%s, test)" % ARM, fontsize=9)
        if "val" in FT and "test" in FT:
            tt = FT["test"].groupby("delta").agg(d=("gt_deleted_frac", "mean"), fr=("false_removed", "sum"), ft=("false_total", "sum"))
            ax[1].plot(tt.index, 100 * tt.fr / tt.ft.clip(lower=1), "-o", label="false components removed (%)", color="#f0883e")
            ax[1].plot(tt.index, 100 * tt.d, "-o", label="true tumour volume deleted (%)", color="#e5534b")
            ax[1].set_xscale("log"); ax[1].set_xlabel("filter size delta (voxels)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.2)
            ax[1].set_title("component-size filter trade-off (test)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "lesion_detection_%s.png" % ARM), dpi=150)
        print("\nwrote", OUT)
    except Exception as e:
        print("(figure skipped: %s)" % e)
    return out


if __name__ == "__main__" or "get_ipython" in globals():
    main()
