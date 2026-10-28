/* payload.js -- the N3 payload adapter (v3, verified against a REAL payload).
 *
 * Resolved conventions (see fixes/real_schema.md, derived empirically from
 * report_BraTS20_Training_095.json and cross-checked against its own meta counts):
 *
 *   vol3d.planes.*  8-bit grayscale PNGs, w = NZ, h = NX*NY; unpack by reshaping the
 *                   pixel stream (row-major) to (NX, NY, NZ) = (X, Y, Z):
 *                       idx(x,y,z) = (x*n[1] + y)*n[2] + z
 *                   (verified: dice 0.96 against pooled full-res GT; byte-identical
 *                   dumps checked by fixes/test_real_payload.js)
 *   mesh3d.layers.* v = uint16 LE xyz triples, pos_cells = v / scale (SIGNED decode wraps
 *                   60% of vertices -- verified against the GT distance transform);
 *                   f = uint16 LE face triples.  Coarse-cell units, same grid as vol3d.n.
 *   slices.*        axial PNG stacks (row=y, col=x), z_full = meta.z_i0 + i:
 *                     flair/t1ce      backgrounds (window/level applies to these ONLY)
 *                     gt/pred/post/err  N3's coloured composites (drawn verbatim)
 *                     gtw/gtc/gte     reference region masks: alpha>0 = full GT region
 *                     pw/pc/pe        DELIVERED region masks: alpha>0 (counts match meta
 *                                     post volumes exactly -- they are not probabilities)
 *                     err composite: red (248,81,73) = EXTRA, blue (31,111,235) = MISSED
 *   ortho.*         coronal (row=z col=x, i0 over full Y), sagittal (row=z col=y, i0 over
 *                   full X); flair + t1ce
 *   meta.export     rle = base64 int32 LE (start,len) pairs over a uint8 volume of shape
 *                   [Z,Y,X] (C-order = NIfTI x-fastest order); NIfTI dims = reversed(shape)
 *   curves          Bayes bound (n/(n+1))*mean + 1/(n+1); lambda* = thresholds[jFn],
 *                   jFn = last index with FN bound <= alpha
 *
 * The pure maths (lambda*, probe, containment) is the verified FIX v2 port; on the axial
 * plane the probe upgrades reference/delivered flags to FULL resolution from the region
 * stacks when they are cached, and always takes p/lambda/unsure from the coarse volume
 * (the only probability source in the payload).
 *
 * Demo payloads written by code/make_demo_case.py (direct byte layers, pos/i/j/k meshes,
 * value-run RLE) are still supported: every section has two decode branches.
 */
(function (root) {
  "use strict";
  var RA = root.RA || (root.RA = {});

  var PLANE_AXES = {                        /* semantic axis indices into (X,Y,Z) */
    axial:    { rows: 1, cols: 0, fix: 2 },
    coronal:  { rows: 2, cols: 0, fix: 1 },
    sagittal: { rows: 2, cols: 1, fix: 0 }
  };
  RA.PLANE_AXES = PLANE_AXES;

  function isNum(a) { return Array.isArray(a) && a.length && typeof a[0] === "number"; }
  function b64ToBytes(s) {
    if (RA.PNG && RA.PNG.b64ToBytes) return RA.PNG.b64ToBytes(s);   /* png.js loaded first in the page */
    var i, out;
    if (typeof atob === "function") {
      var bin = atob(s);
      out = new Uint8Array(bin.length);
      for (i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(s, "base64"));                /* node without png.js */
  }

  /* ------------------------------------------------------------ generic decoders (demo branch) */
  function toU8(v) {
    if (v == null) return null;
    if (v instanceof Uint8Array) return v;
    if (ArrayBuffer.isView(v)) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
    if (typeof v === "object" && !Array.isArray(v)) return toU8(v.data != null ? v.data : v.b64 != null ? v.b64 : v.values);
    if (typeof v === "string") return b64ToBytes(v.replace(/\s/g, ""));
    if (isNum(v)) return new Uint8Array(v);
    return null;
  }
  function toF32(v) {
    if (v == null) return null;
    if (v instanceof Float32Array) return v;
    if (typeof v === "string") {
      var b = b64ToBytes(v.replace(/\s/g, ""));
      return new Float32Array(b.buffer, b.byteOffset, (b.length / 4) | 0);
    }
    if (isNum(v)) {
      if (isNum(v[0])) {
        var flat = new Float32Array(v.length * 3);
        for (var i = 0; i < v.length; i++) { flat[i * 3] = v[i][0]; flat[i * 3 + 1] = v[i][1]; flat[i * 3 + 2] = v[i][2]; }
        return flat;
      }
      return new Float32Array(v);
    }
    if (ArrayBuffer.isView(v)) return new Float32Array(v.buffer, v.byteOffset, (v.byteLength / 4) | 0);
    return null;
  }
  function toImages(v) {
    if (!Array.isArray(v)) return null;
    var out = [], i, s;
    for (i = 0; i < v.length; i++) {
      s = v[i];
      if (typeof s !== "string") { if (s && typeof s.png === "string") s = s.png; else return null; }
      out.push(s.indexOf("data:") === 0 ? s : "data:image/png;base64," + s);
    }
    return out;
  }

  var MESH_STYLE = {
    brain:   { color: "#a8b6c4", opacity: 0.16 },
    gt_wt:   { color: "#3ddc84", opacity: 0.55 },
    gt_tc:   { color: "#2fb86e", opacity: 0.75 },
    gt_et:   { color: "#1f8a52", opacity: 0.95 },
    pred_wt: { color: "#ffa94d", opacity: 0.5 },
    pred_tc: { color: "#e8853a", opacity: 0.7 },
    pred_et: { color: "#c96a2b", opacity: 0.85 },
    post_wt: { color: "#4dabf7", opacity: 0.45 },
    post_tc: { color: "#3b8fd4", opacity: 0.7 },
    post_et: { color: "#2b6cb0", opacity: 0.85 }
  };
  RA.MESH_STYLE = MESH_STYLE;

  /* ------------------------------------------------------------ the loader (async: PNG decoding) */
  async function load(json, pidHint) {
    var report = { sections: [], notes: [] };
    function sec(name, status, detail) { report.sections.push({ name: name, status: status, detail: detail || "" }); }
    function note(s) { report.notes.push(s); }

    var meta = json.meta || {};
    var pid = meta.case_id || pidHint || "(unknown case)";

    var defaults = json.defaults || {};
    var alpha = (typeof defaults.alpha === "number") ? defaults.alpha : 0.10;
    var beta = (typeof defaults.beta === "number") ? defaults.beta : 0.20;
    sec("defaults", json.defaults ? "consumed" : "absent",
        "alpha=" + alpha + ", beta=" + beta + (json.defaults ? "" : " (fallback)"));

    var curves = (json.curves && json.curves.thresholds && json.curves.fn_mean) ? json.curves : null;
    sec("curves", curves ? (curves.fp_mean ? "consumed" : "partial") : "absent",
        curves ? ("n_cal=" + curves.n_cal + ", " + curves.thresholds.length + " thresholds") : "lambda* falls back to meta.lam");

    /* ---- slices: role classification ---- */
    var BG_ROLES = ["flair", "t1ce", "t2", "t1"];
    var COMP_ROLES = { gt: /(^|_)gt$/, ref: /ref|truth/, pred: /^(pred|raw|ai)/, post: /^(post|deliv)/, err: /^(err|diff)/ };
    var REGION_ROLES = ["gtw", "gtc", "gte", "pw", "pc", "pe"];
    var slices = { bgs: {}, comps: {}, regions: {}, rawKeys: [] };
    var rawSlices = json.slices || {};
    Object.keys(rawSlices).forEach(function (k) {
      var imgs = toImages(rawSlices[k]);
      slices.rawKeys.push(k + (imgs ? "(" + imgs.length + ")" : "(?)"));
      if (!imgs) { note("slices." + k + " is not an image stack -- ignored"); return; }
      var lk = k.toLowerCase();
      if (BG_ROLES.indexOf(lk) >= 0) slices.bgs[lk] = imgs;
      else if (REGION_ROLES.indexOf(lk) >= 0) slices.regions[lk] = { uris: imgs, cache: {}, inflight: {} };
      else {
        var role = null;
        Object.keys(COMP_ROLES).some(function (r) { if (COMP_ROLES[r].test(lk)) { role = r; return true; } return false; });
        if (role) slices.comps[role] = slices.comps[role] || imgs;
        else note("slices." + k + " has no assigned role -- not drawn (see Payload map)");
      }
    });
    sec("slices", Object.keys(slices.bgs).length ? "consumed" : "absent",
        "backgrounds: " + (Object.keys(slices.bgs).join(", ") || "NONE") +
        "; composites: " + (Object.keys(slices.comps).join(", ") || "none") +
        "; region masks: " + (Object.keys(slices.regions).join(", ") || "none") +
        "; stacks: " + slices.rawKeys.join(" "));

    /* ---- ortho ---- */
    var ortho = {};
    ["coronal", "sagittal"].forEach(function (pl) {
      var o = (json.ortho || {})[pl] || {};
      var bgs = {};
      BG_ROLES.forEach(function (r) { var im = toImages(o[r]); if (im) bgs[r] = im; });
      ortho[pl] = { bgs: bgs, i0: o.i0 || 0, n: o.n || (bgs.flair ? bgs.flair.length : 0), mid: o.mid };
    });
    sec("ortho", (ortho.coronal.n && ortho.sagittal.n) ? "consumed" : (ortho.coronal.n || ortho.sagittal.n) ? "partial" : "absent",
        "coronal " + ortho.coronal.n + " slices @i0=" + ortho.coronal.i0 +
        ", sagittal " + ortho.sagittal.n + " @i0=" + ortho.sagittal.i0 +
        " (" + Object.keys(ortho.coronal.bgs).join("/") + ")");

    /* ---- vol3d ---- */
    var V3 = { ready: false, layout: null, n: null, full: null, idx: null, layers: {}, found: {}, mri_max: json.vol3d && json.vol3d.mri_max };
    var v3 = json.vol3d || null;
    if (v3 && v3.n && v3.full) {
      V3.n = v3.n.slice(0); V3.full = v3.full.slice(0);
      V3.idx = function (x, y, z) { return ((x * V3.n[1]) + y) * V3.n[2] + z; };
      var need = V3.n[0] * V3.n[1] * V3.n[2];
      if (v3.planes && typeof v3.planes === "object" && typeof v3.planes.wt === "string") {
        /* REAL schema: PNG-packed planes */
        V3.layout = "png-packed (w=NZ, h=NX*NY, row-major -> (X,Y,Z))";
        var names = Object.keys(v3.planes);
        await Promise.all(names.map(async function (k) {
          try {
            var im = await RA.PNG.decode(v3.planes[k]);
            if (im.width * im.height !== need) { note("vol3d.planes." + k + ": " + im.width + "x" + im.height + " != n product " + need); return; }
            V3.layers[k] = (im.channels === 1) ? im.data : await RA.PNG.alphaOf(v3.planes[k]);
            V3.found[k] = "planes." + k + " png " + im.width + "x" + im.height + " ch" + im.channels;
          } catch (e) { note("vol3d.planes." + k + " failed to decode: " + e.message); }
        }));
      } else {
        /* DEMO schema: direct byte layers */
        V3.layout = "direct layers";
        var want = ["wt", "gtw", "tc", "gtc", "et", "gte", "postw", "postc", "poste"];
        Object.keys(v3).forEach(function (k) {
          if (k === "n" || k === "full" || k === "planes" || k === "idx" || k === "mri_max") return;
          var canon = want.indexOf(k) >= 0 ? k : classifyVolLayer(k);
          if (!canon) { note("vol3d." + k + " not recognised -- ignored"); return; }
          var u8 = toU8(v3[k]);
          if (!u8) { note("vol3d." + k + " could not be decoded"); return; }
          if (u8.length !== need) note("vol3d." + k + ": " + u8.length + " bytes, expected " + need);
          V3.layers[canon] = u8; V3.found[canon] = k;
        });
      }
      V3.ready = !!(V3.layers.wt || V3.layers.gtw);
      sec("vol3d", V3.ready ? "consumed" : "absent",
          "n=" + V3.n.join("x") + " full=" + V3.full.join("x") + " · " + V3.layout +
          " · layers: " + Object.keys(V3.layers).join(", "));
      if (V3.layout.indexOf("png") === 0 && !V3.layers.mri) note("vol3d.planes.mri absent (coarse MRI backdrop)");
    } else sec("vol3d", "absent", "no coarse volume -- probe, unsure band and ortho overlays unavailable");

    /* ---- mesh3d ---- */
    var mesh = { layers: {}, order: [], grid: v3 && v3.n ? v3.n.slice(0) : null };
    var m3 = json.mesh3d || null;
    if (m3 && m3.layers) {
      await Promise.all(Object.keys(m3.layers).map(async function (k) {
        var L = m3.layers[k] || {};
        var pos = null, tri = null, enc = "";
        if (typeof L.v === "string" && L.scale) {
          /* REAL schema: uint16 LE triples / scale (signed decode would wrap!) */
          var bytes = b64ToBytes(L.v);
          var u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, (bytes.length / 2) | 0);
          var nv = L.nv || ((u16.length / 3) | 0);
          pos = new Float32Array(nv * 3);
          for (var i = 0; i < nv * 3; i++) pos[i] = u16[i] / L.scale;
          enc = "uint16/scale(" + Math.round(L.scale * 100) / 100 + ")";
          if (L.f) {
            var fb2 = b64ToBytes(L.f);
            var f16 = new Uint16Array(fb2.buffer, fb2.byteOffset, (fb2.length / 2) | 0);
            tri = { i: f16.subarray(0, f16.length - (f16.length % 3)), j: null, k: null, flat: f16 };
          }
        } else {
          pos = toF32(L.pos != null ? L.pos : L.vertices);
          enc = "direct";
          if (L.i && L.j && L.k) tri = { i: toF32(L.i), j: toF32(L.j), k: toF32(L.k) };
        }
        if (!pos || pos.length < 9) { note("mesh3d.layers." + k + ": no usable vertices -- skipped"); return; }
        var name = /brain|head|skull/.test(k.toLowerCase()) ? "brain" : k.toLowerCase().replace(/-/g, "_");
        mesh.layers[name] = { pos: pos, nv: (pos.length / 3) | 0, tri: tri, src: k, enc: enc,
                              style: MESH_STYLE[name] || { color: "#c0c8d0", opacity: 0.5 } };
        mesh.order.push(name);
      }));
      mesh.order.sort(function (a, b) { return a === "brain" ? -1 : b === "brain" ? 1 : a < b ? -1 : 1; });
      sec("mesh3d", mesh.order.length ? "consumed" : "absent",
          (m3.n ? "grid " + m3.n.join("x") + ", " : "") + (m3.cells ? "cells " + m3.cells + ", " : "") +
          mesh.order.length + " layers (" + mesh.order.map(function (nm) { return nm + ":" + mesh.layers[nm].enc; }).join(", ") + ")");
    } else sec("mesh3d", "absent", "the 3D panel will be empty");

    /* ---- export ---- */
    var ex = meta.export || null;
    if (ex && ex.rle) {
      if (typeof ex.rle === "string") {
        var info = decodeRleReal(ex);
        sec("meta.export", info ? "consumed" : "partial",
            info ? ("rle " + info.nRuns + " (start,len) int32 runs over " + ex.shape.join("x") +
                    " (Z,Y,X) uint8 -> " + info.count + " voxels; NIfTI dims " + ex.shape.slice().reverse().join("x"))
                 : "rle string could not be decoded -- .nii button disabled");
      } else {
        var v2 = decodeRleDemo(ex.rle);
        sec("meta.export", v2 ? "consumed" : "partial", v2 ? ("demo value-run rle " + v2.shape.join("x")) : "unrecognised rle shape");
      }
    } else sec("meta.export", "absent", "no export.rle -- the .nii button stays disabled");

    /* ---- everything else captured ---- */
    var extras = {};
    Object.keys(json).forEach(function (k) {
      if (["meta", "defaults", "curves", "slices", "ortho", "vol3d", "mesh3d"].indexOf(k) >= 0) return;
      extras[k] = json[k];
    });
    ["vol_curve", "coverage", "contact", "cohort_png"].forEach(function (k) {
      if (extras[k] != null) sec(k, "consumed", summarise(extras[k]));
      else sec(k, "absent", "optional section not in this payload");
    });
    Object.keys(extras).forEach(function (k) {
      if (["vol_curve", "coverage", "contact", "cohort_png"].indexOf(k) >= 0) return;
      sec(k, "unused", "present but no panel renders it -- " + summarise(extras[k]));
      note("top-level section `" + k + "` is not rendered anywhere");
    });

    var metaUsed = ["case_id", "lam", "gt_volume", "post_volume", "ai", "export", "cert_scope", "n_cal",
                    "ai_review", "thr_tc", "thr_et", "z_i0", "note", "region_rows", "engine_rows",
                    "coverage_rows", "x0", "y0", "z0", "z0_index", "z_full", "n_axial",
                    "filter_feasible", "filter_delta", "vol_wt", "vol_tc", "vol_et",
                    "vol_wt_pred", "vol_tc_pred", "vol_et_pred", "describe", "stratum", "lesion_min_vox"];
    var metaExtra = Object.keys(meta).filter(function (k) { return metaUsed.indexOf(k) < 0; });
    sec("meta", "consumed", "case_id=" + pid +
        (meta.ai_review ? "; ai_review present (FIX v2 payload)" : "; NO ai_review -- payload predates FIX v2") +
        (meta.note ? "; note rendered" : "") + (meta.region_rows ? "; region/engine/coverage rows rendered" : "") +
        (metaExtra.length ? "; extra keys shown in details: " + metaExtra.join(", ") : ""));

    return {
      pid: pid, json: json, meta: meta, defaults: defaults,
      alpha0: alpha, beta0: beta,
      curves: curves, slices: slices, ortho: ortho, V3: V3, mesh: mesh,
      extras: extras, metaExtra: metaExtra, report: report
    };
  }

  function classifyVolLayer(key) {
    var k = key.toLowerCase();
    var region = /et/.test(k) ? "et" : /tc/.test(k) ? "tc" : /wt/.test(k) ? "wt" : null;
    if (!region) return null;
    var kind = /gt|ref|true|truth/.test(k) ? "gt" : /post|deliv|final|frozen/.test(k) ? "post" : "";
    return kind + region;
  }
  function summarise(v) {
    if (Array.isArray(v)) return v.length + " item(s)" + (v.length && typeof v[0] === "object" ? " of {" + Object.keys(v[0]).join(",") + "}" : "");
    if (typeof v === "string") return v.indexOf("data:") === 0 ? "image (" + Math.round(v.length / 1024) + " KB b64)" : "string(" + v.length + ")";
    if (v && typeof v === "object") return "keys: " + Object.keys(v).slice(0, 8).join(", ");
    return String(v);
  }

  /* ------------------------------------------------------------ RLE decoders */
  /* REAL: {shape:[Z,Y,X], dtype:'uint8', n_runs, rle: b64 of int32 LE (start,len) pairs}
     -> Uint8Array in ZYX C-order == NIfTI x-fastest order; niiDims = shape reversed */
  function decodeRleReal(ex) {
    try {
      var b = b64ToBytes(ex.rle);
      var dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
      var shape = ex.shape, total = shape[0] * shape[1] * shape[2];
      var out = new Uint8Array(total);
      var nRuns = ex.n_runs || (dv.byteLength / 8);
      var count = 0;
      for (var r = 0; r < nRuns; r++) {
        var st = dv.getInt32(r * 8, true), ln = dv.getInt32(r * 8 + 4, true);
        if (st < 0 || ln < 0 || st + ln > total) return null;
        out.fill(1, st, st + ln);
        count += ln;
      }
      return { shape: shape, data: out, count: count, nRuns: nRuns, niiDims: [shape[2], shape[1], shape[0]] };
    } catch (e) { return null; }
  }
  /* DEMO: {shape:[X,Y,Z], runs:[value,length,...]} -> reordered to x-fastest for NIfTI */
  function decodeRleDemo(r) {
    try {
      var shape = r.shape, runs = r.runs;
      if (!shape || shape.length !== 3 || !runs) return null;
      var X = shape[0], Y = shape[1], Z = shape[2];
      var src = new Uint8Array(X * Y * Z), o = 0, i;
      for (i = 0; i + 1 < runs.length; i += 2) {
        var v = runs[i] ? 1 : 0, ln = runs[i + 1] | 0;
        if (o + ln > src.length) return null;
        if (v) src.fill(1, o, o + ln);
        o += ln;
      }
      if (o !== src.length) return null;
      var out = new Uint8Array(src.length), x, y, z;
      for (x = 0; x < X; x++) for (y = 0; y < Y; y++) for (z = 0; z < Z; z++) {
        out[x + X * (y + Y * z)] = src[(x * Y + y) * Z + z];      /* XYZ C-order -> x-fastest */
      }
      var count = 0; for (i = 0; i < out.length; i++) count += out[i];
      return { shape: shape, data: out, count: count, nRuns: (runs.length / 2) | 0, niiDims: [X, Y, Z] };
    } catch (e) { return null; }
  }
  function decodeExport(meta) {
    var ex = meta.export;
    if (!ex || !ex.rle) return null;
    return (typeof ex.rle === "string") ? decodeRleReal(ex) : decodeRleDemo(ex.rle);
  }

  /* ------------------------------------------------------------ region alpha cache (full-res probe) */
  function regionPrefetch(model, sliceIdx, radius) {
    var regs = model.slices.regions;
    Object.keys(regs).forEach(function (k) {
      var R = regs[k];
      for (var d = -radius; d <= radius; d++) {
        var i = sliceIdx + d;
        if (i < 0 || i >= R.uris.length || R.cache[i] || R.inflight[i]) continue;
        R.inflight[i] = true;
        RA.PNG.alphaOf(R.uris[i]).then(function (kk, ii) {
          return function (a) { regs[kk].cache[ii] = a; delete regs[kk].inflight[ii]; };
        }(k, i)).catch(function () { });
      }
    });
  }
  function regionPeek(model, key, sliceIdx) {
    var R = model.slices.regions[key];
    return (R && R.cache[sliceIdx]) || null;
  }

  /* ------------------------------------------------------------ lambda* (verified port) */
  function lamFromCurves(curves, alpha, fallback) {
    try {
      var n = curves.n_cal, t = curves.thresholds, K = t.length, jFn = null, i;
      var fb = function (i2) { return (n / (n + 1)) * curves.fn_mean[i2] + 1 / (n + 1); };
      for (i = 0; i < K; i++) { if (fb(i) <= alpha) jFn = i; }
      return t[(jFn !== null) ? jFn : 0];
    } catch (e) { return fallback; }
  }
  function fpBoundAt(curves, alpha) {
    try {
      var n = curves.n_cal, t = curves.thresholds, jFn = null, i;
      var fb = function (i2) { return (n / (n + 1)) * curves.fn_mean[i2] + 1 / (n + 1); };
      for (i = 0; i < t.length; i++) { if (fb(i) <= alpha) jFn = i; }
      if (jFn == null || !curves.fp_mean) return null;
      return (n / (n + 1)) * curves.fp_mean[jFn] + 1 / (n + 1);
    } catch (e) { return null; }
  }

  /* ------------------------------------------------------------ slice state / coarse fi */
  function sliceState(model, plane, state) {
    if (plane === "axial") {
      var bg = model.slices.bgs;
      var nAx = bg[Object.keys(bg)[0]] ? bg[Object.keys(bg)[0]].length : 0;
      return { i: state.i | 0, i0: (model.meta.z_i0 != null ? model.meta.z_i0 : 0),
               n: nAx, full: model.V3.full ? model.V3.full[2] : null };
    }
    var o = model.ortho[plane] || {};
    var first = o.bgs && o.bgs[Object.keys(o.bgs)[0]];
    return { i: state.i | 0, i0: o.i0 || 0, n: o.n || (first ? first.length : 0),
             full: model.V3.full ? model.V3.full[plane === "coronal" ? 1 : 0] : null };
  }
  function coarseFi(model, plane, i) {
    var V3 = model.V3, ax = PLANE_AXES[plane];
    if (!V3 || !V3.full) return 0;
    var s = sliceState(model, plane, { i: i });
    var iAbs = s.i0 + Math.max(0, Math.min(Math.max(0, s.n - 1), s.i | 0));
    return Math.max(0, Math.min(V3.n[ax.fix] - 1, Math.round(iAbs * V3.n[ax.fix] / Math.max(1, s.full))));
  }

  /* ------------------------------------------------------------ cursor probe */
  /* Axial upgrades reference/delivered to FULL resolution when the region alpha planes are
     cached (gtw/gtc/gte, pw/pc/pe); p / lambda / unsure always come from the coarse wt
     volume (the only probability source in the payload).  Ortho planes are fully coarse. */
  function probeAt(model, plane, state, cssX, cssY, el) {
    var V3 = model.V3, meta = model.meta;
    if (!V3 || !V3.ready || !el) return null;
    var ax = PLANE_AXES[plane];
    var s = sliceState(model, plane, state);
    if (!s.n) return null;
    var i = Math.max(0, Math.min(s.n - 1, state.i | 0));
    var lam = model.curves ? lamFromCurves(model.curves, state.alpha, meta.lam) : meta.lam;
    if (lam == null) return null;
    var tTC = (meta.thr_tc != null) ? meta.thr_tc : lam;
    var tET = (meta.thr_et != null) ? meta.thr_et : lam;

    var fullres = false, gw, gc, ge, del, delc, dele, rRep, cRep;
    var regs = model.slices.regions;
    if (plane === "axial" && regs.gtw && regs.pw) {
      var FX = V3.full[0], FY = V3.full[1];
      var fx = Math.floor(cssX / Math.max(1, el.clientWidth) * FX);
      var fy = Math.floor(cssY / Math.max(1, el.clientHeight) * FY);
      if (fx >= 0 && fy >= 0 && fx < FX && fy < FY) {
        var ag = regionPeek(model, "gtw", i), ap = regionPeek(model, "pw", i);
        if (ag && ap) {
          fullres = true;
          var qf = fy * FX + fx;   /* alpha planes are row-major PNG pixels: row=y, col=x */
          gw = ag[qf] > 0;
          var ac = regionPeek(model, "gtc", i), ae = regionPeek(model, "gte", i);
          gc = !!(ac && ac[qf] > 0); ge = !!(ae && ae[qf] > 0);
          del = ap[qf] > 0;
          var apc = regionPeek(model, "pc", i), ape = regionPeek(model, "pe", i);
          delc = !!(apc && apc[qf] > 0); dele = !!(ape && ape[qf] > 0);
          rRep = fy; cRep = fx;
        }
      }
    }
    if (!fullres) {
      var rows = V3.n[ax.rows], cols = V3.n[ax.cols];
      var r = Math.floor(cssY / Math.max(1, el.clientHeight) * rows);
      var c = Math.floor(cssX / Math.max(1, el.clientWidth) * cols);
      if (r < 0 || c < 0 || r >= rows || c >= cols) return null;
      var fi = coarseFi(model, plane, i);
      var cc = [0, 0, 0]; cc[ax.fix] = fi; cc[ax.rows] = r; cc[ax.cols] = c;
      var q = V3.idx(cc[0], cc[1], cc[2]);
      var L = V3.layers;
      var at = function (nm) { var a = L[nm]; return a ? a[Math.min(q, a.length - 1)] : 0; };
      gw = at("gtw") > 127; gc = at("gtc") > 127; ge = at("gte") > 127;
      del = at("postw") > 127; delc = at("postc") > 127; dele = at("poste") > 127;
      rRep = r; cRep = c;
    }
    /* p / AI-at-lambda / unsure: always coarse (the payload's only probability source) */
    var p = 0, pw = false, pc = false, pe = false;
    var tq = coarseQ(model, plane, i, cssX, cssY, el);
    if (tq != null && V3.layers.wt) {
      p = V3.layers.wt[Math.min(tq, V3.layers.wt.length - 1)] / 255;
      pw = p >= lam;
      pc = pw && ((V3.layers.tc ? V3.layers.tc[Math.min(tq, V3.layers.tc.length - 1)] : 0) / 255 >= tTC);
      pe = pc && ((V3.layers.et ? V3.layers.et[Math.min(tq, V3.layers.et.length - 1)] : 0) / 255 >= tET);
    } else if (tq == null && !fullres) {
      return null;                                          /* cursor off the coarse grid */
    }
    return { plane: plane, slice: i + 1, n: s.n, r: rRep, c: cRep, fi: coarseFi(model, plane, i), fullres: fullres,
             gw: !!gw, gc: !!gc, ge: !!ge, pw: !!pw, pc: !!pc, pe: !!pe,
             del: !!del, delc: !!delc, dele: !!dele,
             p: p, lam: lam, unc: Math.abs(p - lam) <= Math.max(0.10, 0.35 * lam) };
  }
  function coarseQ(model, plane, i, cssX, cssY, el) {
    var V3 = model.V3;
    if (!V3 || !V3.ready || !el) return null;
    var ax = PLANE_AXES[plane];
    var rows = V3.n[ax.rows], cols = V3.n[ax.cols];
    var r = Math.floor(cssY / Math.max(1, el.clientHeight) * rows);
    var c = Math.floor(cssX / Math.max(1, el.clientWidth) * cols);
    if (r < 0 || c < 0 || r >= rows || c >= cols) return null;
    var fi = coarseFi(model, plane, i);
    var cc = [0, 0, 0]; cc[ax.fix] = fi; cc[ax.rows] = r; cc[ax.cols] = c;
    return V3.idx(cc[0], cc[1], cc[2]);
  }

  function probeText(q) {
    if (!q) return "";
    var ref = q.ge ? "WT+TC+ET (enhancing core)" : q.gc ? "WT+TC (tumour core)" :
              q.gw ? "WT (oedema/edge)" : "no reference here";
    var ai = q.pe ? "WT+TC+ET" : q.pc ? "WT+TC" : q.pw ? "WT" : "outside";
    var rel = (q.pw && q.gw) ? "agree: inside both outlines" :
              (q.pw && !q.gw) ? "EXTRA: AI only, review first" :
              (!q.pw && q.gw) ? "MISSED: reference only, AI off" :
              "background: inside neither";
    var delTxt = q.del ? (q.dele ? "inside (ET)" : q.delc ? "inside (TC)" : "inside") : "outside";
    return "cursor · " + q.plane + " slice " + q.slice + "/" + q.n + " · row " + q.r + " col " + q.c +
      (q.fullres ? " (full-res)" : "") +
      " · reference: " + ref +
      " · AI at λ=" + Number(q.lam).toFixed(2) + ": " + ai + " (p=" + q.p.toFixed(2) + ", coarse)" +
      " · delivered mask: " + delTxt +
      " · " + rel + (q.unc ? " · IN UNSURE BAND (review purview)" : "");
  }

  /* ------------------------------------------------------------ head containment (verified port) */
  function headCheck(mesh) {
    if (!mesh || !mesh.layers || !mesh.layers.brain) return null;
    var bb = {}, k, i, a;
    for (k in mesh.layers) {
      var L = mesh.layers[k]; if (!L || !L.pos) continue;
      var lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
      for (i = 0; i < L.nv; i++) for (a = 0; a < 3; a++) {
        var v = L.pos[i * 3 + a]; if (v < lo[a]) lo[a] = v; if (v > hi[a]) hi[a] = v;
      }
      bb[k] = [lo, hi];
    }
    var head = bb.brain;
    var bad = [];
    for (k in bb) {
      if (k === "brain") continue;
      var b = bb[k];
      for (a = 0; a < 3; a++) {
        if (b[0][a] < head[0][a] - 1.5 || b[1][a] > head[1][a] + 1.5) { bad.push(k); break; }
      }
    }
    return { head: true, bad: bad };
  }

  RA.load = load;
  RA.lamFromCurves = lamFromCurves;
  RA.fpBoundAt = fpBoundAt;
  RA.probeAt = probeAt;
  RA.probeText = probeText;
  RA.sliceState = sliceState;
  RA.coarseFi = coarseFi;
  RA.coarseQ = coarseQ;
  RA.headCheck = headCheck;
  RA.regionPrefetch = regionPrefetch;
  RA.decodeExport = decodeExport;

  /* meta.describe -> HTML.  Pure (string in, string out) so a node test can run it.
     N3's case_descriptor() builds it from the AI OUTLINE and the scan contrast only, and ships the
     rule behind every line (`why`); this just lays it out.  Everything is escaped. */
  function describeHtml(d) {
    if (!d || typeof d !== "object") return "";
    var esc = function (s) { return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); };
    var tones = { ok: "tone-ok", warn: "tone-warn", info: "tone-info" };
    var items = (d.items || []).map(function (it) {
      return "<li class='" + (tones[it.tone] || "tone-info") + "'><b>" + esc(it.k) + "</b> " + esc(it.v) +
             (it.why ? "<div class='dwhy'>" + esc(it.why) + "</div>" : "") + "</li>";
    }).join("");
    return "<div class='dhead'>" + esc(d.headline) + "</div>" +
           (items ? "<ul class='dlist'>" + items + "</ul>" : "") +
           (d.disclaimer ? "<p class='hint ddisc'>" + esc(d.disclaimer) + "</p>" : "");
  }
  RA.describeHtml = describeHtml;
  RA.fmt = {
    pct: function (v, dg) { return (v == null || isNaN(v)) ? "—" : (100 * v).toFixed(dg == null ? 0 : dg) + " %"; },
    num: function (v, dg) { return (v == null || isNaN(v)) ? "—" : Number(v).toFixed(dg == null ? 2 : dg); },
    cm3: function (mm3) { return (mm3 == null || isNaN(mm3)) ? "—" : (mm3 / 1000).toFixed(2) + " cm³"; },
    pretty: function (k) {
      return k.replace(/_/g, " ").replace(/\b(ai|gt|wt|tc|et|cnr|dice|mm3|inter)\b/gi, function (m) { return m.toUpperCase(); });
    }
  };

  if (typeof module !== "undefined" && module.exports) module.exports = RA;
})(typeof window !== "undefined" ? window : globalThis);
