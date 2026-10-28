#!/usr/bin/env python3
"""Install N3 payloads into the reading-aid app.

The app renders whatever the N3 notebook wrote -- `report_<pid>.json` next to its `.html` page.  To
put REAL cases in the worklist you run N3 on Kaggle (attach the cache and the probs dataset, set
N_REPORT, run the FIX v2 cell), download the `.json` files, and hand them to this script:

    python3 code/install_app_study.py ~/Downloads/report_BraTS20_Training_355.json
    python3 code/install_app_study.py E:\\reports                 (a folder: finds the payloads)
    python3 code/install_app_study.py ~/Downloads/kaggle-output.zip (a zip: reads them out of it)
    python3 code/install_app_study.py --why "the case where TTA turns the corner" report_*.json
    python3 code/install_app_study.py --list
    python3 code/install_app_study.py --remove BraTS20_Training_355

What it does, in order:

  1. checks the payload against the contract the app actually reads (`--strict` to refuse anything
     incomplete).  A payload that is missing a section does not fail loudly in the browser -- it
     just renders a hole, so the check is the point of this script.  v2 additionally inventories
     what each section holds (slice layers, coarse volume layers and encodings, mesh layers with
     or without triangles) because the app's Payload-map drawer shows exactly this;
  2. checks that `meta.case_id` matches the filename, because the app keys everything off the file
     name and the report text off `case_id`: if they disagree, one case's numbers end up under
     another case's name;
  3. copies the payload (and its `.html` page, if you also downloaded it) into `app/data/`;
  4. updates `app/data/index.json`, which is what the worklist reads: label, patient line, and the
     one-line reason.  The label and reason are written from the payload's own numbers unless you
     pass `--tag` / `--why` / `--patient`.

Nothing is ever overwritten silently: an existing payload is refused unless `--force`, and the
previous `index.json` is kept as `index.json.bak`.

Fixes vs the previous installer:
  * re-installing a file that already lives in app/data with --force used to crash with
    shutil.SameFileError -- now detected and skipped cleanly;
  * `meta.ai_review` (FIX v2) is checked: absent payloads install fine but are labelled, because
    the app's AI-review panel says "predates FIX v2" for them;
  * export.rle is shape-checked ({shape:[X,Y,Z], runs:[...]}) so the .nii button does not sit
    disabled without explanation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "app", "data")

# Everything app/static/js reaches for, grouped the way the payload is written.  `required`
# sections must be present for the case to render at all; the optional ones degrade.
CONTRACT = {
    "required": ["meta", "slices", "vol3d", "mesh3d"],
    "optional": ["ortho", "curves", "vol_curve", "coverage", "defaults", "contact"],
}
META_REQUIRED = ["case_id", "lam", "gt_volume", "post_volume", "ai", "export"]
# `alpha` / `beta` live in `defaults` (the operating point the notebook certified), not in `meta`;
# the app reads them through one accessor.  Checked here so a payload that predates that cannot be
# installed and then print "NaN" at a clinician.
DEFAULTS_REQUIRED = ["alpha", "beta"]
# canonical coarse layers the planes/probe/unsure overlay read (payload.js classifies aliases too)
VOL3D_LAYERS = ["wt", "gtw", "tc", "gtc", "et", "gte", "postw", "postc", "poste"]


def fail(msg: str) -> None:
    print("  ERROR  " + msg)


def _png_ihdr(uri: str):
    """(width, height) of a data-URI PNG using only the stdlib, or None."""
    try:
        import base64 as _b64
        import struct as _st
        raw = uri.split(",", 1)[1] if "," in uri else uri
        head = _b64.b64decode(raw[:48])
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        w, h = _st.unpack(">II", head[16:24])
        return (w, h)
    except Exception:
        return None


def _enc_of(v) -> str:
    if isinstance(v, str):
        return "base64" if len(v) % 4 == 0 else "string?"
    if isinstance(v, list):
        return "nested-list" if v and isinstance(v[0], list) else "list"
    return type(v).__name__


def check_payload(pid: str, d: dict, strict: bool) -> list[str]:
    """Returns the problems found.  Empty means the app can render this case completely."""
    bad = []
    for k in CONTRACT["required"]:
        if k not in d:
            bad.append("missing section `%s`" % k)
    m = d.get("meta")
    if isinstance(m, dict):
        for k in META_REQUIRED:
            if k not in m:
                bad.append("missing meta.%s" % k)
        cid = str(m.get("case_id", ""))
        if cid and cid != pid:
            bad.append("meta.case_id is `%s` but the file is `%s.json` -- the app keys the case off "
                       "the file name, so these must agree (rename the file, or re-run N3 for it)" % (cid, pid))
        ai = m.get("ai")
        if isinstance(ai, dict):
            for region in ("WT", "TC", "ET"):
                if region not in ai:
                    bad.append("meta.ai.%s absent (that row of the findings table will be blank)" % region)
        ex = m.get("export")
        if isinstance(ex, dict) and not ex.get("rle"):
            bad.append("meta.export.rle absent -- the .nii export button will be disabled")
        elif isinstance(ex, dict) and isinstance(ex.get("rle"), str):
            # REAL schema: base64 of int32 LE (start,len) pairs over a uint8 (Z,Y,X) volume
            import base64 as _b64
            try:
                nbytes = len(_b64.b64decode(ex["rle"]))
            except Exception:
                nbytes = -1
            shape = ex.get("shape")
            nruns = ex.get("n_runs")
            if nbytes <= 0 or not shape or len(shape) != 3:
                bad.append("meta.export.rle is a string but shape/rle are undecodable -- the .nii "
                           "button stays disabled")
            elif nruns and nbytes != 8 * int(nruns):
                bad.append("meta.export.rle holds %d bytes but n_runs=%s implies %d -- the app cannot "
                           "decode it safely" % (nbytes, nruns, 8 * int(nruns)))
            else:
                print("  note   export.rle: %d (start,len) int32 runs over %s (Z,Y,X) %s"
                      % (nbytes // 8, "x".join(str(t) for t in shape), ex.get("dtype", "?")))
        elif isinstance(ex, dict) and isinstance(ex.get("rle"), dict):
            r = ex["rle"]
            shape = r.get("shape") or r.get("dims") or ex.get("shape")
            runs = r.get("runs") or r.get("data")
            if not shape or not runs:
                bad.append("meta.export.rle is not {shape:[X,Y,Z], runs:[value,length,...]} -- the app "
                           "cannot decode it and the .nii button stays disabled")
        if not (m.get("ai_review") or "").strip():
            print("  note   meta.ai_review absent -- payload predates FIX v2; the app's AI-review "
                  "panel will say so (re-run N3 with the fix cell to get it)")
        for k in ("thr_tc", "thr_et"):
            if m.get(k) is None:
                print("  note   meta.%s absent -- the probe falls back to lambda* for that region" % k)
    defs = d.get("defaults")
    if not isinstance(defs, dict) or any(k not in defs for k in DEFAULTS_REQUIRED):
        bad.append("defaults.alpha / defaults.beta absent -- the certified interval cannot be shown")
    v3 = d.get("vol3d")
    if isinstance(v3, dict):
        if "n" not in v3 or "full" not in v3:
            bad.append("vol3d.n / vol3d.full absent -- the coarse volume cannot be indexed")
        planes = v3.get("planes")
        if isinstance(planes, dict) and planes and isinstance(next(iter(planes.values())), str):
            # REAL schema: PNG-packed planes (w = n[2], h = n[0]*n[1], 8-bit gray)
            need = int(v3.get("n", [0, 0, 0])[0] * v3.get("n", [0, 0, 0])[1] * v3.get("n", [0, 0, 0])[2])
            found = [k for k in VOL3D_LAYERS if k in planes]
            missing = [k for k in VOL3D_LAYERS if k not in planes]
            if not found:
                bad.append("vol3d.planes holds no recognisable layer (expected some of: %s)"
                           % ", ".join(VOL3D_LAYERS))
            dims_ok, dims_bad = [], []
            for k, uri in planes.items():
                wh = _png_ihdr(uri)
                if wh is None:
                    dims_bad.append("%s(not a PNG?)" % k)
                elif wh[0] * wh[1] != need:
                    dims_bad.append("%s(%dx%d != n product %d)" % (k, wh[0], wh[1], need))
                else:
                    dims_ok.append(k)
            if dims_bad:
                bad.append("vol3d.planes size mismatch: %s" % ", ".join(dims_bad))
            print("  note   vol3d: %d PNG-packed planes (%s)%s"
                  % (len(planes), ", ".join(sorted(planes)),
                     "; MISSING %s" % ", ".join(missing) if missing else ""))
        elif planes is not None or any(k in v3 for k in VOL3D_LAYERS):
            # DEMO/simple schema: direct byte layers (+ optional planes axis-map)
            found = [k for k in VOL3D_LAYERS if k in v3]
            missing = [k for k in VOL3D_LAYERS if k not in v3]
            if not found:
                bad.append("vol3d holds no recognisable layer (expected some of: %s) -- probe, unsure "
                           "band and ortho overlays unavailable" % ", ".join(VOL3D_LAYERS))
            else:
                encs = sorted({_enc_of(v3[k]) for k in found})
                print("  note   vol3d layers %s (encoding: %s)%s"
                      % (", ".join(found), "/".join(encs),
                         "; MISSING %s" % ", ".join(missing) if missing else ""))
        else:
            bad.append("vol3d.planes empty -- the planes and the 3D view have nothing to draw")
    mk = d.get("mesh3d")
    if isinstance(mk, dict):
        layers = mk.get("layers") or {}
        if not layers:
            bad.append("mesh3d.layers empty -- the 3D panel will be blank")
        else:
            no_pos, no_tri, u16layers = [], [], []
            for k, L in layers.items():
                if not isinstance(L, dict):
                    no_pos.append(k); continue
                if L.get("v") and L.get("scale"):
                    u16layers.append(k)                       # real schema: uint16/scale
                    if not L.get("f"):
                        no_tri.append(k)
                elif L.get("pos") is not None or L.get("vertices") is not None:
                    if not (L.get("i") and L.get("j") and L.get("k")):
                        no_tri.append(k)
                else:
                    no_pos.append(k)
            if no_pos:
                bad.append("mesh3d layers without v/pos: %s -- they cannot be drawn at all"
                           % ", ".join(no_pos))
            print("  note   mesh3d layers: %s (%s)%s"
                  % (", ".join(sorted(layers)),
                     "uint16/scale" if u16layers else "direct pos",
                     "" if not no_tri else "  (no faces, drawn as point clouds: %s)" % ", ".join(sorted(no_tri))))
            if not any("brain" in k.lower() or "head" in k.lower() for k in layers):
                bad.append("mesh3d has no brain/head layer -- the head-containment check cannot run "
                           "and the 3D view has no anatomy context")
    sl = d.get("slices")
    if isinstance(sl, dict):
        if not any(k.lower() in ("flair", "t1ce", "t2", "t1") or "flair" in k.lower() for k in sl):
            bad.append("slices has no background stack (a key like `flair`) -- the axial plane has "
                       "nothing to window/level")
        else:
            print("  note   slices stacks: %s" % ", ".join("%s(%d)" % (k, len(v) if isinstance(v, list) else -1)
                                                           for k, v in sl.items()))
    ort = d.get("ortho")
    if isinstance(ort, dict):
        miss = [p for p in ("coronal", "sagittal") if not (ort.get(p) or {}).get("flair")]
        if miss:
            print("  note   ortho missing %s -- those planes fall back to coarse overlays only"
                  % ", ".join(miss))
        else:
            print("  note   ortho: coronal %d @i0=%s, sagittal %d @i0=%s (+%s)"
                  % (len(ort["coronal"]["flair"]), ort["coronal"].get("i0"),
                     len(ort["sagittal"]["flair"]), ort["sagittal"].get("i0"),
                     ", ".join(k for k in ort["coronal"] if k not in ("flair", "i0", "n", "mid")) or "flair only"))
    for k in CONTRACT["optional"]:
        if k not in d:
            print("  note   no `%s` section: that part of the page falls back, which is fine" % k)
    return bad


def auto_why(d: dict) -> str:
    """One line, built only from numbers the run produced."""
    m = d.get("meta", {})
    ai = (m.get("ai") or {}).get("WT") or {}
    bits = []
    scope = (m.get("cert_scope") or "").strip()
    bits.append("held-out case" if not scope else "held-out case (%s certificate)" % scope)
    if m.get("lam") is not None:
        bits.append("λ* %s" % ("%.2f" % float(m["lam"])))
    if ai.get("agree") is not None:
        bits.append("WT Dice %.2f" % float(ai["agree"]))
    if ai.get("covered") is not None:
        bits.append("covered %d %%" % round(100 * float(ai["covered"])))
    if ai.get("extra") is not None:
        bits.append("extra %d %%" % round(100 * float(ai["extra"])))
    return " · ".join(bits)


def auto_tag(d: dict) -> str:
    scope = (d.get("meta", {}).get("cert_scope") or "").strip()
    return ("real case · %s" % scope) if scope else "real case"


def load_index(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print("  ERROR  %s is not valid JSON (%s); fix or delete it before installing" % (path, e))
            sys.exit(2)
    return {}


def save_index(path: str, man: dict) -> None:
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    json.dump(man, open(path, "w", encoding="utf-8"), indent=1, sort_keys=True)


def cmd_list(man: dict) -> None:
    if not man:
        print("  the worklist manifest is empty")
        return
    print("  %-26s %-22s %s" % ("case", "label", "why"))
    for pid in sorted(man):
        e = man[pid] or {}
        print("  %-26s %-22s %s" % (pid, e.get("tag") or e.get("variant") or "(no label)",
                                    e.get("why") or ""))


def expand_inputs(given: list[str]) -> tuple[list[str], list[str]]:
    """Turn whatever the user typed into a concrete list of `report_*.json` files.

    Three shapes arrive in practice, all of them reasonable, and an installer that only accepts one of
    them is a dead end:

      * the files themselves -- `report_001.json`, or a wildcard the shell already expanded;
      * a FOLDER they were downloaded into -- e.g. `E:\\reports`.  Look for `report_*.json`, then for
        any `*.json`, first at the top level and then one level down (Kaggle's download of a whole
        version's output often lands in a subfolder);
      * the ZIP Kaggle hands out for that output -- read the payloads (and any paired `.html`) straight
        out of it, so nobody has to unzip anything by hand.

    Returns (paths, notes).  Files pulled out of a zip are written to a temporary folder that lives for
    the duration of the run; the installer then treats them exactly like downloaded files, including
    finding the `.html` next to them.
    """
    files: list[str] = []
    notes: list[str] = []
    tmps: list[str] = []

    def from_dir(d: str) -> list[str]:
        # Both the top level and one level down, deduplicated by case: downloads of two different
        # Kaggle versions often sit in the same folder, and a nested copy that is silently skipped
        # looks exactly like a case that failed to install.
        top = sorted(glob.glob(os.path.join(d, "report_*.json")))
        deep = sorted(glob.glob(os.path.join(d, "*", "report_*.json")))
        picks, seen = [], {}
        for f in top + deep:
            key = os.path.basename(f)
            if key in seen:
                notes.append("%s: also found at %s -- using the shallower copy"
                             % (key, os.path.relpath(f, d)))
                continue
            seen[key] = f
            picks.append(f)
        if picks:
            if deep and not top:
                notes.append("%s: found them one level down, in %s" % (d, os.path.dirname(deep[0])))
            return picks
        loose = sorted(glob.glob(os.path.join(d, "*.json")))
        loose = [f for f in loose if os.path.basename(f) != "index.json"]
        if loose:
            notes.append("%s: no report_*.json, using the %d .json file(s) there" % (d, len(loose)))
            return loose
        pages = sorted(glob.glob(os.path.join(d, "*.html")))
        if pages:
            notes.append("%s holds %d .html page(s) but no payload. The app reads the "
                         "report_<pid>.json files -- the .html is the standalone page and is optional. "
                         "Download the Output of the N3 version again and take the .json files."
                         % (d, len(pages)))
            return []
        zips = sorted(glob.glob(os.path.join(d, "*.zip")))
        if zips:
            notes.append("%s: no JSON on disk, reading %s" % (d, os.path.basename(zips[0])))
            return from_zip(zips[0])
        return []

    def from_zip(z: str) -> list[str]:
        try:
            zf = zipfile.ZipFile(z)
        except Exception as e:
            notes.append("%s is not a readable zip: %s" % (os.path.basename(z), e))
            return []
        members = [m for m in zf.namelist() if os.path.basename(m).startswith("report_") and
                   m.endswith(".json")]
        if not members:
            members = [m for m in zf.namelist() if m.endswith(".json") and
                       os.path.basename(m) != "index.json"]
        if not members:
            notes.append("%s contains no report_*.json (%d file(s) in it)" % (os.path.basename(z), len(zf.namelist())))
            return []
        tmp = tempfile.mkdtemp(prefix="n3payloads_")
        tmps.append(tmp)
        out = []
        for m in members:
            base = os.path.basename(m)
            dst = os.path.join(tmp, base)
            with zf.open(m) as src, open(dst, "wb") as fh:
                shutil.copyfileobj(src, fh)
            out.append(dst)
            html = os.path.splitext(m)[0] + ".html"
            if html in zf.namelist():                      # the paired report page, if the zip has it
                with zf.open(html) as src, open(os.path.join(tmp, os.path.basename(html)), "wb") as fh:
                    shutil.copyfileobj(src, fh)
        notes.append("%s: %d payload(s) read out of the zip" % (os.path.basename(z), len(out)))
        return sorted(out)

    for a in given:
        if os.path.isdir(a):
            found = from_dir(a)
            if found:
                notes.append("%s: %d payload(s)" % (a, len(found)))
                files.extend(found)
            else:
                notes.append("%s is a folder with nothing to install -- no report_*.json, no *.json, "
                             "no .zip in it (did the download finish?)" % a)
            continue
        if os.path.isfile(a):
            if a.lower().endswith(".zip"):
                files.extend(from_zip(a))
            elif a.lower().endswith(".html"):
                twin = os.path.splitext(a)[0] + ".json"
                notes.append("%s is the standalone report page, not a payload. The app reads %s "
                             "-- %s" % (os.path.basename(a), os.path.basename(twin),
                                        "it is next to it" if os.path.exists(twin)
                                        else "download it from the same N3 version's Output"))
            else:
                files.append(a)
            continue
        hit = sorted(glob.glob(a))                    # cmd.exe does not expand wildcards for us
        hit = [h for h in hit if not os.path.isdir(h)]
        if hit:
            files.extend(hit)
        else:
            notes.append("no such file or folder: %s" % a)
    return files, notes


def install_one(path: str, args, man: dict) -> bool:
    base = os.path.basename(path)
    pid = base[len("report_"):-len(".json")] if base.startswith("report_") else base[:-5]
    try:
        d = json.load(open(path, encoding="utf-8"))
    except IsADirectoryError:
        fail("%s is a folder, not a payload. Pass the folder itself and this script will find the "
             "report_*.json inside it." % base)
        return False
    except PermissionError as e:
        fail("%s cannot be read (%s). If it is a folder, pass the folder instead of opening it; if it "
             "is one file, check that it finished downloading." % (base, e.strerror or e))
        return False
    except Exception as e:
        fail("%s is not valid JSON: %s" % (base, e))
        return False
    if not isinstance(d.get("meta"), dict):
        fail("%s does not look like an N3 payload (no `meta` object) -- is this a stray JSON file?"
             % base)
        return False

    print("  %s" % pid)
    problems = check_payload(pid, d, args.strict)
    mismatched = [p for p in problems if "must agree" in p]
    holes = [p for p in problems if p not in mismatched]
    for p in problems:
        print("       - " + p)
    if mismatched and not args.force:
        # This one is never "install anyway": every panel is keyed off the file name while the report
        # sentence is keyed off meta.case_id, so a mismatch prints one case's numbers under another
        # case's name.  --force is the only way past it.
        fail("%s refused: the payload's case_id and its file name disagree. Rename the file to "
             "match, re-run N3 for it, or pass --force if you know which is right." % pid)
        return False
    if holes:
        if args.strict:
            fail("%s refused (--strict): fix the run, or install without --strict to accept a "
                 "partly-rendering case" % pid)
            return False
        print("       installing anyway; the app will show a hole where a section is missing")

    dst_json = os.path.join(args.data, pid + ".json")
    if os.path.exists(dst_json) and not args.force:
        fail("%s is already installed. Use --force to replace it, or --remove first." % pid)
        return False
    html_src = os.path.join(os.path.dirname(os.path.abspath(path)), "report_" + pid + ".html")
    if not os.path.exists(html_src):
        alt = os.path.join(os.path.dirname(os.path.abspath(path)), pid + ".html")
        html_src = alt if os.path.exists(alt) else None

    if args.dry_run:
        print("       would write %s%s" % (dst_json, " and its .html page" if html_src else " (no .html next to it)"))
        print("       would label it: %s | %s" % (args.tag or auto_tag(d), args.why or auto_why(d)))
        return True

    os.makedirs(args.data, exist_ok=True)
    # FIX: reinstalling a payload that already sits in app/data (e.g. pointing the installer at the
    # data dir itself with --force) used to die on shutil.SameFileError.
    if not os.path.exists(dst_json) or not os.path.samefile(os.path.abspath(path), dst_json):
        shutil.copyfile(path, dst_json)
    else:
        print("       source already is %s -- left in place" % dst_json)
    if html_src and not (os.path.exists(os.path.join(args.data, pid + ".html")) and
                         os.path.samefile(html_src, os.path.join(args.data, pid + ".html"))):
        shutil.copyfile(html_src, os.path.join(args.data, pid + ".html"))
    entry = dict(man.get(pid) or {})
    entry["patient"] = args.patient or entry.get("patient") or pid
    entry["tag"] = args.tag or entry.get("tag") or auto_tag(d)
    entry["why"] = args.why or entry.get("why") or auto_why(d)
    entry["source"] = "N3 payload installed from " + base
    entry.pop("variant", None)          # a real case has no phantom variant
    man[pid] = entry
    print("       installed %s  [%s]" % (pid, entry["tag"]))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Install N3 payloads into the reading-aid app.")
    ap.add_argument("files", nargs="*", help="report_<pid>.json downloaded from the N3 run")
    ap.add_argument("--data", default=DATA, help="payload directory (default: app/data)")
    ap.add_argument("--list", action="store_true", help="show the worklist and exit")
    ap.add_argument("--remove", metavar="PID", help="uninstall a case (payload, page and manifest line)")
    ap.add_argument("--tag", help="worklist label for this case (default: built from the payload)")
    ap.add_argument("--why", help="the one-line reason this case is in the set")
    ap.add_argument("--patient", help="what the worklist shows on the second line")
    ap.add_argument("--force", action="store_true", help="replace an already-installed case")
    ap.add_argument("--strict", action="store_true", help="refuse payloads that cannot render fully")
    ap.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    args = ap.parse_args()

    index_path = os.path.join(args.data, "index.json")
    man = load_index(index_path)

    if args.list:
        cmd_list(man)
        return 0

    if args.remove:
        pid = args.remove
        gone = []
        for suffix in (".json", ".html"):
            f = os.path.join(args.data, pid + suffix)
            if os.path.exists(f) and not args.dry_run:
                os.remove(f)
                gone.append(os.path.basename(f))
            elif os.path.exists(f):
                gone.append(os.path.basename(f) + " (would remove)")
        if pid in man:
            man.pop(pid)
            if not args.dry_run:
                save_index(index_path, man)
            gone.append("index.json entry")
        print("  removed " + (", ".join(gone) if gone else "nothing (no such case)"))
        return 0

    files, notes = expand_inputs(args.files)
    for n in notes:
        print("  " + n)
    if not files:
        if not notes:
            ap.print_help()
        print("\n  nothing installed. Point this script at the report_*.json files, at the folder they "
              "are in, or at the zip Kaggle gave you.")
        return 1
    print("  installing %d payload(s):" % len(files))

    ok = [install_one(f, args, man) for f in files]
    if not args.dry_run and any(ok):
        save_index(index_path, man)
        print("\n  restart the server to see the new worklist:")
        print("    python3 app/server.py --port 8000")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
