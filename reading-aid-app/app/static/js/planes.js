/* planes.js -- the 2D plane painter (v3, real-payload schema).
 *
 * FIX behaviours are native here:
 *   1. window/level touches the SCAN ONLY (background PNG); every overlay draws unfiltered.
 *   2. the unsure-band dither paints on EVERY plane at the live lambda (coarse volume).
 *   3. pointer moves produce the probe readout (payload.js probeAt/probeText): on the axial
 *      plane reference/delivered come FULL-RESOLUTION from the region stacks (gtw/gtc/gte,
 *      pw/pc/pe alpha>0) once decoded; p/lambda/unsure always come from the coarse volume,
 *      which is the payload's only probability source (labelled "coarse" in the readout).
 *
 * Overlay sources:
 *   axial     composites from N3 PNG stacks (gt green contour, pred orange @lambda*, post blue,
 *             err red=EXTRA/blue=MISSED) + live coarse AI@lambda + coarse unsure dither
 *   coronal/  everything computed from the coarse volume (the payload has no ortho mask PNGs)
 *   sagittal
 */
(function (root) {
  "use strict";
  var RA = root.RA || (root.RA = {});

  var COLOR = {
    gt: "#3ddc84", raw: "#ffa94d", post: "#4dabf7",
    errExtra: [248, 81, 73], errMiss: [31, 111, 235],
    unc: [219, 228, 238]
  };
  RA.PLANE_COLORS = COLOR;

  /* ------------------------------------------------------------ image loading (browser paint path) */
  var cache = {};
  function imgOf(uri, cb) {
    if (typeof Image === "undefined") return null;             /* node harness: skip painting */
    var im = cache[uri];
    if (im && im.__ready) return im;
    if (!im) {
      im = cache[uri] = new Image();
      im.onload = function () { im.__ready = true; if (im.__cb) { var f = im.__cb; im.__cb = null; f(); } };
      im.src = uri;
    }
    im.__cb = cb || null;
    return im.__ready ? im : null;
  }
  RA.preload = function (uris, around, radius) {
    for (var j = -radius; j <= radius; j++) {
      var i = around + j;
      if (i >= 0 && i < uris.length) imgOf(uris[i], null);
    }
  };

  /* ------------------------------------------------------------ coarse overlay builder */
  /* One ImageData (cols x rows of the coarse plane) with the requested layers, at lambda.
     wants = {gt, pred, live, post, err, unc}.  pred uses meta.lam (frozen lambda*),
     live uses the slider lambda.  Same colour language as N3's own composites. */
  function coarseOverlay(model, plane, fi, lam, wants) {
    var V3 = model.V3;
    if (!V3 || !V3.ready || typeof document === "undefined") return null;
    var ax = RA.PLANE_AXES[plane];
    var cols = V3.n[ax.cols], rows = V3.n[ax.rows];
    var cv = coarseOverlay._cv || (coarseOverlay._cv = document.createElement("canvas"));
    cv.width = cols; cv.height = rows;
    var ctx = cv.getContext("2d");
    var im = ctx.createImageData(cols, rows);
    var d = im.data, L = V3.layers, meta = model.meta;
    var lamStar = meta.lam;
    var tTC = (meta.thr_tc != null) ? meta.thr_tc : lam;
    var tET = (meta.thr_et != null) ? meta.thr_et : lam;
    var band = Math.max(0.10, 0.35 * lam);
    var r, c, cc, q, p, gw, del, set;
    for (r = 0; r < rows; r++) {
      for (c = 0; c < cols; c++) {
        cc = [0, 0, 0]; cc[ax.fix] = fi; cc[ax.rows] = r; cc[ax.cols] = c;
        q = V3.idx(cc[0], cc[1], cc[2]);
        if (q >= (L.wt ? L.wt.length : 0)) continue;
        p = L.wt ? L.wt[q] / 255 : 0;
        gw = L.gtw && L.gtw[q] > 127;
        del = (L.postw && L.postw[q] > 127) || (L.postc && L.postc[q] > 127) || (L.poste && L.poste[q] > 127);
        var o = (r * cols + c) * 4;
        /* paint order: unsure dither -> err -> post -> pred/live -> gt on top */
        if (wants.unc && Math.abs(p - lam) <= band) {
          if ((r + c) & 1) set = [COLOR.unc[0], COLOR.unc[1], COLOR.unc[2], 235];
          else continue;
        } else if (wants.err && ((gw && !del) || (!gw && del))) {
          set = (gw && !del) ? [COLOR.errMiss[0], COLOR.errMiss[1], COLOR.errMiss[2], 210]
                             : [COLOR.errExtra[0], COLOR.errExtra[1], COLOR.errExtra[2], 210];
        } else if (wants.post && del) {
          set = (L.poste && L.poste[q] > 127) ? [43, 108, 176, 215]
              : (L.postc && L.postc[q] > 127) ? [59, 143, 212, 185] : [77, 171, 247, 145];
        } else if (wants.live && p >= lam) {
          set = (L.et && L.tc && p >= lam && L.tc[q] / 255 >= tTC && L.et[q] / 255 >= tET) ? [201, 106, 43, 215]
              : (L.tc && L.tc[q] / 255 >= tTC) ? [232, 133, 58, 185] : [255, 169, 77, 145];
        } else if (wants.pred && lamStar != null && p >= lamStar) {
          set = [240, 136, 62, 110];
        } else if (wants.gt && gw) {
          set = (L.gte && L.gte[q] > 127) ? [31, 138, 82, 220]
              : (L.gtc && L.gtc[q] > 127) ? [47, 184, 110, 190] : [61, 220, 132, 140];
        } else continue;
        d[o] = set[0]; d[o + 1] = set[1]; d[o + 2] = set[2]; d[o + 3] = set[3];
      }
    }
    ctx.putImageData(im, 0, 0);
    return cv;
  }
  RA.coarseOverlay = coarseOverlay;

  /* ------------------------------------------------------------ the painter */
  function create(canvas, model, plane, initialState) {
    var ctx = (typeof canvas.getContext === "function") ? canvas.getContext("2d") : null;
    var bgKeys = plane === "axial" ? Object.keys(model.slices.bgs)
                                   : Object.keys((model.ortho[plane] || {}).bgs || {});
    var st = Object.assign({
      i: 0, alpha: model.alpha0, beta: model.beta0, win: 1, lvl: 1,
      bg: bgKeys.indexOf("flair") >= 0 ? "flair" : (bgKeys[0] || null),
      wants: { gt: true, pred: true, live: false, post: true, err: false, unc: false }
    }, initialState || {});
    var probeEl = null, lastPt = null, attached = false;

    function bgUris() {
      if (!st.bg) return [];
      return plane === "axial" ? (model.slices.bgs[st.bg] || [])
                               : (((model.ortho[plane] || {}).bgs || {})[st.bg] || []);
    }
    function compUris(role) {
      return plane === "axial" ? (model.slices.comps[role] || null) : null;
    }
    function lam() {
      return model.curves ? RA.lamFromCurves(model.curves, st.alpha, model.meta.lam) : model.meta.lam;
    }

    function draw() {
      if (!ctx) return;
      var W = canvas.width, H = canvas.height;
      ctx.save();
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0b0f14";
      ctx.fillRect(0, 0, W, H);

      /* background: the ONLY layer window/level may touch */
      var uris = bgUris();
      var i = Math.max(0, Math.min(Math.max(0, uris.length - 1), st.i | 0));
      var im = uris.length ? imgOf(uris[i], function () { draw(); }) : null;
      ctx.filter = (st.win !== 1 || st.lvl !== 1) ? ("contrast(" + st.win + ") brightness(" + st.lvl + ")") : "none";
      ctx.imageSmoothingEnabled = true;
      if (im) ctx.drawImage(im, 0, 0, W, H);
      ctx.filter = "none";

      /* N3 composites (axial only): verbatim, unfiltered */
      ctx.imageSmoothingEnabled = false;
      [["gt", "gt"], ["pred", "pred"], ["post", "post"], ["err", "err"]].forEach(function (pr) {
        if (!st.wants[pr[0]]) return;
        var stack = compUris(pr[1]);
        if (!stack) return;
        var m = imgOf(stack[Math.min(i, stack.length - 1)], function () { draw(); });
        if (m) ctx.drawImage(m, 0, 0, W, H);
      });

      /* coarse layers: everything on ortho planes; live-AI + unsure on axial */
      var L = lam();
      if (L != null && model.V3.ready) {
        var fi = RA.coarseFi(model, plane, st.i);
        var wants = {
          gt: st.wants.gt && plane !== "axial",
          pred: st.wants.pred && plane !== "axial",
          live: !!st.wants.live,
          post: st.wants.post && plane !== "axial",
          err: st.wants.err && plane !== "axial",
          unc: !!st.wants.unc
        };
        if (wants.gt || wants.pred || wants.live || wants.post || wants.err || wants.unc) {
          var ov = coarseOverlay(model, plane, fi, L, wants);
          if (ov) ctx.drawImage(ov, 0, 0, W, H);
        }
      }
      ctx.imageSmoothingEnabled = true;
      ctx.restore();
      refreshProbe();
    }

    function refreshProbe() {
      if (!probeEl) return;
      var q = lastPt ? RA.probeAt(model, plane, st, lastPt.x, lastPt.y, canvas) : null;
      probeEl.textContent = q ? RA.probeText(q)
        : "cursor · move over the image — reference vs AI-at-λ vs delivered vs unsure band";
      probeEl.classList.toggle("hot", !!(q && (q.unc || (q.gw !== q.pw) || (!!q.del !== !!q.gw))));
    }
    function attachProbe(lineEl) {
      probeEl = lineEl;
      if (attached || typeof canvas.addEventListener !== "function") return;
      attached = true;
      canvas.style.cursor = "crosshair";
      canvas.addEventListener("pointermove", function (ev) {
        var rect = canvas.getBoundingClientRect();
        lastPt = { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
        refreshProbe();
      });
      canvas.addEventListener("pointerleave", function () { lastPt = null; refreshProbe(); });
    }

    return { canvas: canvas, ctx: ctx, plane: plane, state: st, draw: draw,
             attachProbe: attachProbe, lam: lam, bgUris: bgUris, bgKeys: bgKeys,
             setSlice: function (i) { st.i = i; draw(); } };
  }
  RA.Planes = { create: create, imgOf: imgOf };
})(typeof window !== "undefined" ? window : globalThis);
