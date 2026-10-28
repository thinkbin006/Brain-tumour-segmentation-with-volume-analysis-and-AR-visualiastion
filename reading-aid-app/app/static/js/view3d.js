/* view3d.js -- the 3D panel (v4): a readable head + tumour onion, still 100% payload-fed.
 *
 * Why v4 exists (diagnosed in fixes/diag_brain.js + headless-chrome shots):
 *  - N3's mesh3d "brain" layer is an isosurface of the COARSE INTENSITY MAP, which is a
 *    tissue ribbon (88% of the coarse grid is zero even inside the head) -> the shell
 *    arrives as 17 disconnected patches ("patchy, not a whole brain").
 *  - pred_wt/pred_tc are VERTEX-IDENTICAL to post_wt/post_tc in the real payload, and the
 *    gt/post/pred families nest inside each other; stacked translucent shells multiply
 *    their alphas (0.85^3 ~ 2% transmission) and coincident surfaces z-fight, so plotly
 *    showed one orange blob and nothing else.
 * v4 therefore:
 *  - rebuilds a whole head shell IN-APP as a visual hull of the three FULL-RES background
 *    silhouette stacks (axial 64 + coronal 28 + sagittal 28, Otsu per slice, largest
 *    component, hole-filled), intersected on the coarse grid, cuberilled with MERGED
 *    vertices and Taubin-smoothed;
 *  - cuts away the camera-side hemisphere of every tumour shell ("cutaway") so each line
 *    of sight crosses at most one wall per family -> nested rings read as onion layers
 *    instead of multiplying alphas;
 *  - hides the pred family by default where it is vertex-identical to post (checkbox to
 *    inspect), and keeps N3's original patchy brain mesh as an opt-in comparison trace.
 * Vertices still decode uint16/scale (the FIX v2.1 meshFix); every NUMBER elsewhere in the
 * app remains full-resolution.  Coarse surfaces are visual aids only.
 */
(function (root) {
  "use strict";
  var RA = root.RA || (root.RA = {});

  var liveTimer = null;

  /* ---------------------------------------------------------- merged cuberille ----------
     One quad per exposed face of {vol >= thr}, corners MERGED across voxels (v3 emitted 8
     duplicate verts per voxel: 180k tris for a 15k-voxel surface).  Returns {pos,tri}. */
  function cuberilleVol(at, nx, ny, nz, thr) {
    var pos = [], tri = [], vmap = new Map();
    function vid(x, y, z) {           /* corner at half-integer lattice (x-.5+cx, ...) */
      var k = (x * 2 + 1024) * 4194304 + (y * 2 + 1024) * 2048 + (z * 2 + 1024);
      var i = vmap.get(k);
      if (i === undefined) { i = pos.length / 3; vmap.set(k, i); pos.push(x, y, z); }
      return i;
    }
    var C = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
    var F = [[0,1,2,3],[7,6,5,4],[0,1,5,4],[3,2,6,7],[1,2,6,5],[0,3,7,4]];
    function voxel(x, y, z) {
      var v = [0,1,2,3,4,5,6,7].map(function (ci) {
        return vid(x - 0.5 + C[ci][0], y - 0.5 + C[ci][1], z - 0.5 + C[ci][2]);
      });
      for (var fi = 0; fi < 6; fi++) {
        var f = F[fi];
        tri.push(v[f[0]], v[f[1]], v[f[2]], v[f[0]], v[f[2]], v[f[3]]);
      }
    }
    for (var x = 0; x < nx; x++) for (var y = 0; y < ny; y++) for (var z = 0; z < nz; z++) {
      if (at(x, y, z) < thr) continue;
      if ((x > 0 ? at(x - 1, y, z) : 0) < thr || (x < nx - 1 ? at(x + 1, y, z) : 0) < thr ||
          (y > 0 ? at(x, y - 1, z) : 0) < thr || (y < ny - 1 ? at(x, y + 1, z) : 0) < thr ||
          (z > 0 ? at(x, y, z - 1) : 0) < thr || (z < nz - 1 ? at(x, y, z + 1) : 0) < thr) voxel(x, y, z);
    }
    if (!tri.length) return null;
    return { pos: pos, tri: tri };
  }
  function cuberille(V3, lam) {       /* live AI surface: coarse probability volume */
    var L = V3.layers.wt;
    if (!L) return null;
    var nx = V3.n[0], ny = V3.n[1], nz = V3.n[2], thr = Math.round(lam * 255);
    return cuberilleVol(function (x, y, z) {
      if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
      return L[((x * ny) + y) * nz + z];
    }, nx, ny, nz, thr);
  }
  RA.cuberille = cuberille;

  /* ---------------------------------------------------------- Taubin smoothing ---------- */
  function smooth(pos, tri, iters, lam, mu) {
    var nv = (pos.length / 3) | 0;
    var adj = new Array(nv);
    for (var t = 0; t < tri.length; t += 3) {
      for (var e = 0; e < 3; e++) {
        var a = tri[t + e], b = tri[t + (e + 1) % 3];
        (adj[a] || (adj[a] = [])).push(b);
        (adj[b] || (adj[b] = [])).push(a);
      }
    }
    var p = pos.slice();
    for (var it = 0; it < iters; it++) {
      var step = (it % 2 === 0) ? lam : mu;
      var q = new Array(p.length);
      for (var v = 0; v < nv; v++) {
        var nb = adj[v];
        if (!nb || !nb.length) { q[v*3] = p[v*3]; q[v*3+1] = p[v*3+1]; q[v*3+2] = p[v*3+2]; continue; }
        var sx = 0, sy = 0, sz = 0;
        for (var k = 0; k < nb.length; k++) { sx += p[nb[k]*3]; sy += p[nb[k]*3+1]; sz += p[nb[k]*3+2]; }
        sx /= nb.length; sy /= nb.length; sz /= nb.length;
        q[v*3]   = p[v*3]   + step * (sx - p[v*3]);
        q[v*3+1] = p[v*3+1] + step * (sy - p[v*3+1]);
        q[v*3+2] = p[v*3+2] + step * (sz - p[v*3+2]);
      }
      p = q;
    }
    return p;
  }

  /* ---------------------------------------------------------- 2D silhouettes ------------ */
  function otsu(gray) {
    var h = new Array(256).fill(0), i;
    for (i = 0; i < gray.length; i++) h[gray[i]]++;
    var tot = gray.length, sum = 0;
    for (i = 0; i < 256; i++) sum += i * h[i];
    var sb = 0, wb = 0, best = -1, thr = 127;
    for (i = 0; i < 256; i++) {
      wb += h[i]; if (wb === tot) break;
      sb += i * h[i];
      var mb = sb / wb, mf = (sum - sb) / (tot - wb);
      var between = wb * (tot - wb) * (mb - mf) * (mb - mf);
      if (between > best) { best = between; thr = i; }
    }
    return thr;
  }
  function sil2d(gray, w, h) {
    /* N3 backgrounds sit on a FLAT per-slice grey floor (87..135 in the real payload) and
       tissue deviates from it in BOTH directions (dark CSF/sulci, bright GM/tumour), so
       Otsu (which splits tissue-vs-tissue when background is the minority) is wrong here:
       silhouette = |v - median| > T, then a 2D closing to seal partial-volume borders. */
    var sorted = Uint8Array.from(gray).sort();
    var med = sorted[(sorted.length / 2) | 0];
    var T = 12;
    var m = new Uint8Array(w * h), i;
    for (i = 0; i < m.length; i++) m[i] = Math.abs(gray[i] - med) > T ? 1 : 0;
    /* 2D closing: dilate then erode (4-neighbourhood) */
    var d = Uint8Array.from(m);
    for (var y0 = 0; y0 < h; y0++) for (var x0 = 0; x0 < w; x0++) {
      var q0 = y0 * w + x0; if (m[q0]) continue;
      if ((x0 > 0 && m[q0 - 1]) || (x0 < w - 1 && m[q0 + 1]) || (y0 > 0 && m[q0 - w]) || (y0 < h - 1 && m[q0 + w])) d[q0] = 1;
    }
    var e = Uint8Array.from(d);
    for (var y1 = 0; y1 < h; y1++) for (var x1 = 0; x1 < w; x1++) {
      var q1 = y1 * w + x1; if (!d[q1]) continue;
      if (!((x1 > 0 && d[q1 - 1]) && (x1 < w - 1 && d[q1 + 1]) && (y1 > 0 && d[q1 - w]) && (y1 < h - 1 && d[q1 + w]))) e[q1] = 0;
    }
    m = e;
    /* largest 4-connected component */
    var lab = new Int32Array(m.length), nl = 0, st = new Int32Array(m.length), bestN = 0, bestMark = null;
    for (var s = 0; s < m.length; s++) {
      if (!m[s] || lab[s]) continue;
      nl++; var top = 0; st[top++] = s; lab[s] = nl; var mark = [];
      while (top) {
        var q = st[--top]; mark.push(q);
        var x = q % w, y = (q / w) | 0;
        if (x > 0 && m[q - 1] && !lab[q - 1]) { lab[q - 1] = nl; st[top++] = q - 1; }
        if (x < w - 1 && m[q + 1] && !lab[q + 1]) { lab[q + 1] = nl; st[top++] = q + 1; }
        if (y > 0 && m[q - w] && !lab[q - w]) { lab[q - w] = nl; st[top++] = q - w; }
        if (y < h - 1 && m[q + w] && !lab[q + w]) { lab[q + w] = nl; st[top++] = q + w; }
      }
      if (mark.length > bestN) { bestN = mark.length; bestMark = mark; }
    }
    var out = new Uint8Array(w * h);
    if (!bestMark) return out;
    for (i = 0; i < bestMark.length; i++) out[bestMark[i]] = 1;
    /* fill holes: flood background from the border */
    var bg = new Uint8Array(w * h); top = 0;
    var push = function (q) { if (!bg[q] && !out[q]) { bg[q] = 1; st[top++] = q; } };
    for (var x2 = 0; x2 < w; x2++) { push(x2); push((h - 1) * w + x2); }
    for (var y2 = 0; y2 < h; y2++) { push(y2 * w); push(y2 * w + w - 1); }
    while (top) {
      var q2 = st[--top], xx = q2 % w, yy = (q2 / w) | 0;
      if (xx > 0) push(q2 - 1); if (xx < w - 1) push(q2 + 1);
      if (yy > 0) push(q2 - w); if (yy < h - 1) push(q2 + w);
    }
    for (i = 0; i < out.length; i++) if (!out[i] && !bg[i]) out[i] = 1;
    return out;
  }

  /* ---------------------------------------------------------- head shell (visual hull) -- */
  var shellCache = new WeakMap();
  function headShell(model) {
    if (shellCache.has(model)) return shellCache.get(model);
    var pr = buildShell(model).catch(function (e) {
      console.warn("[view3d] head shell rebuild failed, falling back to N3 brain mesh:", e);
      return null;
    });
    shellCache.set(model, pr);
    return pr;
  }
  function bgStack(model, which) {
    var bgs = which === "axial" ? model.slices.bgs
            : ((model.ortho[which] || {}).bgs || {});
    return bgs.flair || bgs.t1ce || bgs[Object.keys(bgs)[0]] || null;
  }
  async function buildShell(model) {
    var ax = bgStack(model, "axial"), co = bgStack(model, "coronal"), sa = bgStack(model, "sagittal");
    if (!ax || !co || !sa) return null;
    var dec = [], i;
    for (i = 0; i < ax.length; i++) dec.push(RA.PNG.decode(ax[i]));
    for (i = 0; i < co.length; i++) dec.push(RA.PNG.decode(co[i]));
    for (i = 0; i < sa.length; i++) dec.push(RA.PNG.decode(sa[i]));
    var ims = await Promise.all(dec);
    var nAx = ax.length, nCo = co.length;
    var AW = ims[0].width, AH = ims[0].height;                 /* axial: row=y col=x */
    var CW = ims[nAx].width, CH = ims[nAx].height;            /* coronal: row=z col=x */
    var SW = ims[nAx + nCo].width, SH = ims[nAx + nCo].height;/* sagittal: row=z col=y */
    var silA = [], silC = [], silS = [];
    var uniA = new Uint8Array(AW * AH), uniC = new Uint8Array(CW * CH), uniS = new Uint8Array(SW * SH);
    for (i = 0; i < nAx; i++) {
      var s = sil2d(ims[i].data, AW, AH); silA.push(s);
      for (var q = 0; q < s.length; q++) if (s[q]) uniA[q] = 1;
    }
    for (i = 0; i < nCo; i++) {
      var s2 = sil2d(ims[nAx + i].data, CW, CH); silC.push(s2);
      for (var q2 = 0; q2 < s2.length; q2++) if (s2[q2]) uniC[q2] = 1;
    }
    for (i = 0; i < sa.length; i++) {
      var s3 = sil2d(ims[nAx + nCo + i].data, SW, SH); silS.push(s3);
      for (var q3 = 0; q3 < s3.length; q3++) if (s3[q3]) uniS[q3] = 1;
    }
    /* intersect the three silhouette extrusions on the COARSE grid */
    var V3 = model.V3, n = V3.n, full = V3.full;
    var nx = n[0], ny = n[1], nz = n[2], FX = full[0], FY = full[1], FZ = full[2];
    var meta = model.meta || {};
    var zi0 = meta.z_i0 != null ? meta.z_i0 : 0;
    var ci0 = (model.ortho.coronal || {}).i0 || 0, si0 = (model.ortho.sagittal || {}).i0 || 0;
    var mask = new Uint8Array(nx * ny * nz);
    for (var x = 0; x < nx; x++) {
      var fx = Math.min(FX - 1, Math.floor((x + 0.5) * FX / nx));
      for (var y = 0; y < ny; y++) {
        var fy = Math.min(FY - 1, Math.floor((y + 0.5) * FY / ny));
        for (var z = 0; z < nz; z++) {
          var fz = Math.min(FZ - 1, Math.floor((z + 0.5) * FZ / nz));
          var ia = (fz >= zi0 && fz < zi0 + nAx) ? silA[fz - zi0] : null;
          var A = ia ? ia[fy * AW + fx] : uniA[fy * AW + fx];
          if (!A) continue;
          var ic = (fy >= ci0 && fy < ci0 + nCo) ? silC[fy - ci0] : null;
          var C = ic ? ic[fz * CW + fx] : uniC[fz * CW + fx];
          if (!C) continue;
          var isg = (fx >= si0 && fx < si0 + silS.length) ? silS[fx - si0] : null;
          var S = isg ? isg[fz * SW + fy] : uniS[fz * SW + fy];
          if (!S) continue;
          mask[(x * ny + y) * nz + z] = 1;
        }
      }
    }
    /* largest 6-connected component + hole fill + 2 closing passes (seals silhouette
       notches where a view's dark anatomy, e.g. the frontal base, dropped out) */
    mask = largest6(mask, nx, ny, nz);
    mask = fill3(mask, nx, ny, nz);
    for (var ci = 0; ci < 2; ci++) mask = close3(mask, nx, ny, nz);
    /* surface at HALF the coarse resolution: the shell is a backdrop visual and a
       full-res cuberille of a 36k-voxel surface would cost ~220k tris per react() */
    var hx = (nx + 1) >> 1, hy = (ny + 1) >> 1, hz = (nz + 1) >> 1;
    var half = new Uint8Array(hx * hy * hz);
    for (var x3 = 0; x3 < hx; x3++) for (var y3 = 0; y3 < hy; y3++) for (var z3 = 0; z3 < hz; z3++) {
      var on = 0;
      for (var dx = 0; dx < 2 && !on; dx++) for (var dy = 0; dy < 2 && !on; dy++) for (var dz = 0; dz < 2; dz++) {
        var X = x3 * 2 + dx, Y = y3 * 2 + dy, Z = z3 * 2 + dz;
        if (X < nx && Y < ny && Z < nz && mask[(X * ny + Y) * nz + Z]) on = 1;
      }
      half[(x3 * hy + y3) * hz + z3] = on;
    }
    var cb = cuberilleVol(function (x, y, z) {
      if (x < 0 || y < 0 || z < 0 || x >= hx || y >= hy || z >= hz) return 0;
      return half[(x * hy + y) * hz + z];
    }, hx, hy, hz, 1);
    if (!cb) return null;
    for (var pi = 0; pi < cb.pos.length; pi += 3) {
      cb.pos[pi] = cb.pos[pi] * 2 - 0.5; cb.pos[pi+1] = cb.pos[pi+1] * 2 - 0.5; cb.pos[pi+2] = cb.pos[pi+2] * 2 - 0.5;
    }
    cb.pos = smooth(cb.pos, cb.tri, 16, 0.55, -0.58);
    return { pos: cb.pos, tri: cb.tri, nv: (cb.pos.length / 3) | 0, nf: (cb.tri.length / 3) | 0,
             mask: mask, n: [nx, ny, nz] };
  }
  function largest6(m, nx, ny, nz) {
    var lab = new Int32Array(m.length), st = new Int32Array(m.length), best = 0, mark = null;
    var idx = function (x, y, z) { return (x * ny + y) * nz + z; };
    for (var s = 0; s < m.length; s++) {
      if (!m[s] || lab[s]) continue;
      var top = 0; st[top++] = s; lab[s] = 1; var cur = [];
      while (top) {
        var q = st[--top]; cur.push(q);
        var z = q % nz, y = ((q - z) / nz) % ny, x = ((q - z) / nz - y) / ny;
        var nb = [x > 0 ? idx(x-1,y,z) : -1, x < nx-1 ? idx(x+1,y,z) : -1, y > 0 ? idx(x,y-1,z) : -1,
                  y < ny-1 ? idx(x,y+1,z) : -1, z > 0 ? idx(x,y,z-1) : -1, z < nz-1 ? idx(x,y,z+1) : -1];
        for (var k = 0; k < 6; k++) if (nb[k] >= 0 && m[nb[k]] && !lab[nb[k]]) { lab[nb[k]] = 1; st[top++] = nb[k]; }
      }
      if (cur.length > best) { best = cur.length; mark = cur; }
    }
    var out = new Uint8Array(m.length);
    if (mark) for (var i = 0; i < mark.length; i++) out[mark[i]] = 1;
    return out;
  }
  function close3(m, nx, ny, nz) {
    var idx = function (x, y, z) { return (x * ny + y) * nz + z; };
    var d = Uint8Array.from(m), x, y, z, q;
    for (x = 0; x < nx; x++) for (y = 0; y < ny; y++) for (z = 0; z < nz; z++) {
      q = idx(x, y, z); if (m[q]) continue;
      if ((x > 0 && m[idx(x-1,y,z)]) || (x < nx-1 && m[idx(x+1,y,z)]) || (y > 0 && m[idx(x,y-1,z)]) ||
          (y < ny-1 && m[idx(x,y+1,z)]) || (z > 0 && m[idx(x,y,z-1)]) || (z < nz-1 && m[idx(x,y,z+1)])) d[q] = 1;
    }
    var e = Uint8Array.from(d);
    for (x = 0; x < nx; x++) for (y = 0; y < ny; y++) for (z = 0; z < nz; z++) {
      q = idx(x, y, z); if (!d[q]) continue;
      if (!((x > 0 && d[idx(x-1,y,z)]) && (x < nx-1 && d[idx(x+1,y,z)]) && (y > 0 && d[idx(x,y-1,z)]) &&
            (y < ny-1 && d[idx(x,y+1,z)]) && (z > 0 && d[idx(x,y,z-1)]) && (z < nz-1 && d[idx(x,y,z+1)]))) e[q] = 0;
    }
    return e;
  }
  function fill3(m, nx, ny, nz) {
    var idx = function (x, y, z) { return (x * ny + y) * nz + z; };
    var bg = new Uint8Array(m.length), st = new Int32Array(m.length), top = 0;
    var push = function (q) { if (!bg[q] && !m[q]) { bg[q] = 1; st[top++] = q; } };
    for (var x = 0; x < nx; x++) for (var y = 0; y < ny; y++) { push(idx(x,y,0)); push(idx(x,y,nz-1)); }
    for (var x2 = 0; x2 < nx; x2++) for (var z = 0; z < nz; z++) { push(idx(x2,0,z)); push(idx(x2,ny-1,z)); }
    for (var y2 = 0; y2 < ny; y2++) for (var z2 = 0; z2 < nz; z2++) { push(idx(0,y2,z2)); push(idx(nx-1,y2,z2)); }
    while (top) {
      var q = st[--top];
      var zz = q % nz, yy = ((q - zz) / nz) % ny, xx = ((q - zz) / nz - yy) / ny;
      var nb = [xx > 0 ? idx(xx-1,yy,zz) : -1, xx < nx-1 ? idx(xx+1,yy,zz) : -1, yy > 0 ? idx(xx,yy-1,zz) : -1,
                yy < ny-1 ? idx(xx,yy+1,zz) : -1, zz > 0 ? idx(xx,yy,zz-1) : -1, zz < nz-1 ? idx(xx,yy,zz+1) : -1];
      for (var k = 0; k < 6; k++) if (nb[k] >= 0) push(nb[k]);
    }
    var out = Uint8Array.from(m);
    for (var i = 0; i < out.length; i++) if (!m[i] && !bg[i]) out[i] = 1;
    return out;
  }
  RA.headShell = headShell;

  /* ---------------------------------------------------------- trace styling ------------- */
  var STYLE = {
    gt_et:  { color: "#1f8a52", opacity: 0.95 }, gt_tc:  { color: "#2fb86e", opacity: 0.80 },
    gt_wt:  { color: "#3ddc84", opacity: 0.50 },
    post_et:{ color: "#2b6cb0", opacity: 0.88 }, post_tc:{ color: "#3b8fd4", opacity: 0.68 },
    post_wt:{ color: "#4dabf7", opacity: 0.42 },
    pred_et:{ color: "#c96a2b", opacity: 0.85 }, pred_tc:{ color: "#e8853a", opacity: 0.66 },
    pred_wt:{ color: "#ffa94d", opacity: 0.40 },
    live:   { color: "#ffd8a8", opacity: 0.35 },
    shell:  { color: "#9db2c4", opacity: 0.10, flat: false },
    n3brain:{ color: "#6b7683", opacity: 0.15 }
  };
  var CLIP_FAMILIES = { gt: 1, post: 1, pred: 1 };   /* cutaway applies to tumour shells */

  function clipCentre(mesh) {
    var L = mesh.layers.gt_wt || mesh.layers.post_wt || mesh.layers.pred_wt;
    if (!L) return null;
    var mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9];
    for (var i = 0; i < L.pos.length; i += 3) for (var a = 0; a < 3; a++) {
      mn[a] = Math.min(mn[a], L.pos[i+a]); mx[a] = Math.max(mx[a], L.pos[i+a]);
    }
    return [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
  }
  var CAM_EYE = [1.5, -1.7, 0.8];
  function clipTri(pos, tri, c) {
    var dl = Math.hypot(CAM_EYE[0], CAM_EYE[1], CAM_EYE[2]);
    var d = [CAM_EYE[0]/dl, CAM_EYE[1]/dl, CAM_EYE[2]/dl];
    var out = [];
    for (var t = 0; t < tri.length; t += 3) {
      var cx = 0, cy = 0, cz = 0;
      for (var e = 0; e < 3; e++) { cx += pos[tri[t+e]*3]; cy += pos[tri[t+e]*3+1]; cz += pos[tri[t+e]*3+2]; }
      cx /= 3; cy /= 3; cz /= 3;
      if ((cx-c[0])*d[0] + (cy-c[1])*d[1] + (cz-c[2])*d[2] <= 0) out.push(tri[t], tri[t+1], tri[t+2]);
    }
    return out;
  }

  function meshTrace(name, pos, tri, st, extraName) {
    var x = [], y = [], z = [], i;
    for (i = 0; i < pos.length; i += 3) { x.push(pos[i]); y.push(pos[i+1]); z.push(pos[i+2]); }
    var ii = [], jj = [], kk = [];
    for (i = 0; i < tri.length; i += 3) { ii.push(tri[i]); jj.push(tri[i+1]); kk.push(tri[i+2]); }
    return { type: "mesh3d", name: extraName || name, x: x, y: y, z: z, i: ii, j: jj, k: kk,
             color: st.color, opacity: st.opacity, flatshading: st.flat !== false,
             lighting: st.flat === false ? { ambient: 0.55, diffuse: 0.5, specular: 0.08, roughness: 0.9 } : undefined,
             hoverinfo: "name", showlegend: true };
  }
  function traceFor(name, L, clipC) {
    var tri = (L.tri && L.tri.flat) ? L.tri.flat : (L.tri && L.tri.i ? interleave(L.tri) : null);
    if (!tri) return null;
    var pos = L.pos;
    if (clipC) tri = clipTri(pos, tri, clipC);
    if (!tri.length) return null;
    return meshTrace(name, pos, tri, STYLE[name] || { color: "#c0c8d0", opacity: 0.5 });
  }
  function interleave(t) {
    var out = new Array(t.i.length * 3);
    for (var i = 0; i < t.i.length; i++) { out[i*3] = t.i[i]; out[i*3+1] = t.j[i]; out[i*3+2] = t.k[i]; }
    return out;
  }

  var current = null;   /* {div, model, warnEl, lam, showLive, vis, shell} */
  var DEFAULT_VIS = { shell: true, cut: true, gt: true, post: true, pred: false, n3: false };

  function render(div, model, warnEl, opts) {
    opts = opts || {};
    current = { div: div, model: model, warnEl: warnEl,
                lam: opts.lam != null ? opts.lam : model.meta.lam,
                showLive: !!opts.showLive, vis: Object.assign({}, DEFAULT_VIS, opts.vis || {}),
                shell: null };
    if (typeof root.Plotly === "undefined") {
      div.innerHTML = "<div class='empty3d'>plotly.min.js did not load — it is vendored under /static/vendor/</div>";
      return Promise.resolve(null);
    }
    var mesh = model.mesh;
    if (!mesh || !mesh.order.length) {
      div.innerHTML = "<div class='empty3d'>no mesh3d.layers in this payload — the 3D panel has nothing to draw (see Payload map)</div>";
      return Promise.resolve(null);
    }
    noteText();
    redraw();
    return headShell(model).then(function (sh) {
      if (!current || current.model !== model) return sh;
      current.shell = sh;
      if (sh && current.vis.shell) redraw();
      noteText();
      return sh;
    });
  }
  function noteText() {
    var w = current && current.warnEl;
    if (!w) return;
    var model = current.model, mesh = model.mesh;
    var bits = [];
    if (current.shell) bits.push("head shell rebuilt in-app from full-res silhouettes (visual hull, " +
      current.shell.nf.toLocaleString() + " tris)");
    else bits.push("no background stacks — head shell unavailable");
    if (current.vis.cut) bits.push("tumour shells cut away on the camera side so the nested rings read");
    var hc = mesh.layers.brain ? RA.headCheck(mesh) : null;
    if (hc && hc.bad.length) bits.push("⚠ N3 head shell does not enclose: " + hc.bad.join(", "));
    bits.push("coarse surfaces are visual aids — every NUMBER is full-resolution");
    w.className = (hc && hc.bad.length) ? "warnblk" : "hint";
    w.textContent = bits.join(" · ");
  }

  function redraw() {
    if (!current || typeof root.Plotly === "undefined") return;
    var model = current.model, mesh = model.mesh, vis = current.vis;
    var clipC = vis.cut ? clipCentre(mesh) : null;
    var data = [];
    /* onion order: innermost first per family, families gt -> post -> pred; then live, shell */
    ["gt_et", "gt_tc", "gt_wt"].forEach(function (n2) {
      if (vis.gt && mesh.layers[n2]) { var t = traceFor(n2, mesh.layers[n2], clipC); if (t) data.push(t); }
    });
    ["post_et", "post_tc", "post_wt"].forEach(function (n2) {
      if (vis.post && mesh.layers[n2]) { var t = traceFor(n2, mesh.layers[n2], clipC); if (t) data.push(t); }
    });
    ["pred_et", "pred_tc", "pred_wt"].forEach(function (n2) {
      if (vis.pred && mesh.layers[n2]) { var t = traceFor(n2, mesh.layers[n2], clipC); if (t) data.push(t); }
    });
    if (current.showLive && model.V3.ready && current.lam != null) {
      var cb = cuberille(model.V3, current.lam);
      if (cb) {
        var tri = cb.tri;
        if (clipC) tri = clipTri(cb.pos, tri, clipC);
        if (tri.length) data.push(meshTrace("live", cb.pos, tri, STYLE.live,
          "AI@λ=" + Number(current.lam).toFixed(2) + " (live, coarse)"));
      }
    }
    if (vis.n3 && mesh.layers.brain) {
      var t2 = traceFor("n3brain", mesh.layers.brain, null);
      if (t2) data.push(t2);
    }
    if (vis.shell && current.shell) {
      data.push(meshTrace("shell", current.shell.pos, current.shell.tri, STYLE.shell,
                        "head shell (rebuilt, full-res silhouettes)"));
    }
    if (!data.length) {
      current.div.innerHTML = "<div class='empty3d'>nothing selected — tick a 3D layer</div>";
      return;
    }
    var layout = {
      paper_bgcolor: "rgba(0,0,0,0)", margin: { l: 0, r: 0, t: 4, b: 4 },
      font: { size: 9.5, color: "#9fb0c0" },
      scene: {
        aspectmode: "data",
        xaxis: { visible: false }, yaxis: { visible: false }, zaxis: { visible: false },
        camera: { eye: { x: CAM_EYE[0], y: CAM_EYE[1], z: CAM_EYE[2] }, up: { x: 0, y: 0, z: 1 } }
      },
      legend: { orientation: "h", y: -0.02, x: 0, bgcolor: "rgba(0,0,0,0)",
                font: { size: 9, color: "#9fb0c0" } }
    };
    root.Plotly.react(current.div, data, layout, { responsive: true, displayModeBar: false });
  }

  function setLambda(lam, showLive) {
    if (!current) return;
    current.lam = lam;
    if (showLive != null) current.showLive = showLive;
    if (liveTimer) clearTimeout(liveTimer);
    liveTimer = setTimeout(redraw, 180);
  }
  function setVisible(vis) {
    if (!current) return;
    var changed = false;
    Object.keys(vis || {}).forEach(function (k) {
      if (vis[k] != null && current.vis[k] !== vis[k]) { current.vis[k] = vis[k]; changed = true; }
    });
    if (changed) { redraw(); noteText(); }
  }
  function resize() {
    if (current && root.Plotly && root.Plotly.Plots && root.Plotly.Plots.resize) {
      try { root.Plotly.Plots.resize(current.div); } catch (e) { }
    }
  }

  RA.View3D = { render: render, setLambda: setLambda, setVisible: setVisible, resize: resize,
                redraw: redraw, cuberille: cuberille, headShell: headShell, smooth: smooth,
                DEFAULT_VIS: DEFAULT_VIS };
})(typeof window !== "undefined" ? window : globalThis);
