/* app.js -- workstation wiring (v3, real-payload schema).
 * Everything shown comes from the N3 payload; the only recomputation is the live
 * lambda-star, probe and overlay maths, ported verbatim from the verified FIX v2 page overrides. */
(function (root) {
  "use strict";
  var RA = root.RA || (root.RA = {});
  var $ = function (id) { return document.getElementById(id); };

  var model = null;
  var painters = {};
  var state = { alpha: 0.10, beta: 0.20, win: 1, lvl: 1 };
  var wants = { gt: true, pred: true, live: false, post: true, err: false, unc: false };

  /* ------------------------------------------------------------ worklist */
  function fmtMini(s) {
    var bits = [];
    if (s.lam != null) bits.push("λ* " + Number(s.lam).toFixed(2));
    if (s.dice != null) bits.push("Dice " + Number(s.dice).toFixed(2));
    if (s.covered != null) bits.push("cov " + Math.round(100 * s.covered) + "%");
    if (s.extra != null) bits.push("ext " + Math.round(100 * s.extra) + "%");
    return bits.join(" · ");
  }
  function loadWorklist() {
    fetch("api/studies").then(function (r) { return r.json(); }).then(function (studies) {
      var wl = $("worklist");
      wl.innerHTML = "";
      if (!studies.length) {
        wl.innerHTML = "<div class='empty3d'>no studies installed yet —<br>python3 code/install_app_study.py report_&lt;pid&gt;.json</div>";
        return;
      }
      studies.forEach(function (s) {
        var card = document.createElement("div");
        card.className = "case";
        card.dataset.pid = s.pid;
        card.innerHTML =
          "<div class='case-top'><span class='pid'>" + s.pid + "</span>" +
          (s.has_review ? "<span class='chip chip-ok' title='payload carries meta.ai_review (FIX v2)'>AI✓</span>"
                        : "<span class='chip chip-warn' title='payload predates FIX v2: no meta.ai_review'>AI—</span>") + "</div>" +
          "<div class='case-tag'>" + (s.tag || "study") + "</div>" +
          (s.why ? "<div class='case-why'>" + s.why + "</div>" : "") +
          "<div class='case-nums'>" + fmtMini(s) + "</div>" +
          "<div class='case-bottom'>" + (s.patient && s.patient !== s.pid ? "<span>" + s.patient + "</span>" : "") +
          (s.has_report ? "<a href='report/" + encodeURIComponent(s.pid) + "' target='_blank' rel='noopener'>report ↗</a>" : "") +
          /* AR */
          (s.has_ar ? " <a href='AR/" + encodeURIComponent(s.pid) + "' target='_blank' rel='noopener' " +
                      "onclick='event.stopPropagation()' title='Open this case in browser AR (phone)'>AR ↗</a>" : "") +
          "</div>";
        card.addEventListener("click", function () { selectCase(s.pid, card); });
        wl.appendChild(card);
      });
      if (studies.length === 1) selectCase(studies[0].pid, wl.firstChild);
    }).catch(function (e) {
      $("worklist").innerHTML = "<div class='empty3d'>/api/studies failed: " + e + "</div>";
    });
  }

  /* ------------------------------------------------------------ case boot */
  function selectCase(pid, card) {
    document.querySelectorAll(".case").forEach(function (c) { c.classList.remove("sel"); });
    if (card) card.classList.add("sel");
    $("status").textContent = "loading " + pid + " … (decoding coarse PNG volumes)";
    fetch("api/study/" + encodeURIComponent(pid)).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (json) {
      return RA.load(json, pid);
    }).then(function (m) {
      model = m;
      state.alpha = m.alpha0; state.beta = m.beta0;
      m.__state = state;
      bootCase();
    }).catch(function (e) {
      $("status").textContent = "could not load " + pid + ": " + (e && e.message || e);
      console.error(e);
    });
  }

  function bootCase() {
    var m = model, meta = m.meta;
    $("caseChip").textContent = m.pid;
    $("dataChip").textContent = (meta.cert_scope ? "certificate scope: " + meta.cert_scope : "no certificate scope") +
      " · n_cal " + (meta.n_cal != null ? meta.n_cal : "?");
    /* a low-contrast scan whose own stratum is not certifiable carries the POOLED certificate: an average-case range,
       not a guarantee for this scan.  Needs meta.stratum (N3 final); older payloads simply show no warning. */
    var pooledOnLow = meta.cert_scope === "pooled" && meta.stratum === "low";
    if (pooledOnLow) {
      $("dataChip").textContent += " · low contrast: pooled interval, not certified for this scan";
      $("dataChip").title = "This scan's contrast level has no certificate of its own at this α/β; the pooled certificate is an average over all scans. See the note under 'certificate' for how often it held on held-out low-contrast scans.";
    }
    $("dataChip").classList.toggle("chip-warn", pooledOnLow);
    var kpis = [
      ["λ*", meta.lam != null ? Number(meta.lam).toFixed(2) : "—"],
      ["reference WT", RA.fmt.cm3(meta.gt_volume)],
      ["delivered WT", RA.fmt.cm3(meta.post_volume)]
    ];
    if (meta.vol_wt) kpis.push(["vol WT (truth)", String(meta.vol_wt).replace(" mm³", "")]);
    $("kpis").innerHTML = kpis.map(function (kv) {
      return "<div class='kpi'><span>" + kv[0] + "</span><b>" + kv[1] + "</b></div>";
    }).join("");

    /* AI review (FIX v2 field; this real payload predates it -> honest notice) */
    var ai = $("aiReview");
    if (meta.describe && RA.describeHtml) {            /* structured card, rule shown under every line */
      ai.className = "ai-review describe";
      ai.innerHTML = "<b>Automatic description</b> <span class='hint'>(rule-based, from the AI outline — not a diagnosis)</span>" +
        RA.describeHtml(meta.describe);
    } else if (meta.ai_review) {
      ai.className = "ai-review";
      ai.innerHTML = "<b>AI review</b> <span class='hint'>(rule-based impression from counted volumes — not a diagnosis)</span><p>" +
        String(meta.ai_review).replace(/</g, "&lt;") + "</p>";
    } else {
      ai.className = "ai-review missing";
      ai.innerHTML = "<b>AI review</b><p class='hint'>this payload predates FIX v2 (no <code>meta.ai_review</code>). " +
        "Re-run N3 with the fix cell (fixes/N3_fix_cell.py) to get the impression line — the app will then show it here.</p>";
    }

    /* findings: meta.ai, every region, every field */
    var ft = $("findings");
    var aiRows = meta.ai || {};
    var cols = [];
    Object.keys(aiRows).forEach(function (rg) {
      Object.keys(aiRows[rg] || {}).forEach(function (k) { if (cols.indexOf(k) < 0) cols.push(k); });
    });
    if (cols.length) {
      ft.innerHTML = "<table class='kv'><tr><td>region</td>" +
        cols.map(function (c) { return "<td>" + RA.fmt.pretty(c) + "</td>"; }).join("") + "</tr>" +
        ["WT", "TC", "ET"].filter(function (rg) { return aiRows[rg]; }).map(function (rg) {
          return "<tr><td><b>" + rg + "</b></td>" + cols.map(function (c) {
            var v = (aiRows[rg] || {})[c];
            if (v == null) return "<td class='num'>—</td>";
            if (typeof v === "number") {
              if (/covered|extra|agree/.test(c) && Math.abs(v) <= 1) return "<td class='num'>" + RA.fmt.pct(v, 1) + "</td>";
              if (/mm3/.test(c)) return "<td class='num'>" + v.toLocaleString() + "</td>";
              return "<td class='num'>" + RA.fmt.num(v, 3) + "</td>";
            }
            return "<td class='num'>" + String(v) + "</td>";
          }).join("") + "</tr>";
        }).join("") + "</table>";
    } else ft.innerHTML = "<div class='empty3d'>meta.ai absent</div>";

    /* background channel buttons (flair / t1ce) */
    var bgBox = $("bgSel");
    bgBox.innerHTML = "";
    var bgKeys = Object.keys(m.slices.bgs);
    bgKeys.forEach(function (k) {
      var b = document.createElement("button");
      b.className = "tgl" + (k === "flair" ? " on" : "");
      b.textContent = k.toUpperCase();
      b.addEventListener("click", function () {
        Object.keys(painters).forEach(function (p) { painters[p].state.bg = k; });
        Array.prototype.forEach.call(bgBox.children, function (c2) { c2.classList.remove("on"); });
        b.classList.add("on");
        redrawAll();
      });
      bgBox.appendChild(b);
    });
    if (!bgKeys.length) bgBox.innerHTML = "<span class='hint'>no background stacks</span>";

    /* painters */
    var nAx = bgKeys.length ? m.slices.bgs[bgKeys[0]].length : 1;
    $("zSlider").max = Math.max(0, nAx - 1);
    $("cySlider").max = Math.max(0, (m.ortho.coronal.n || 1) - 1);
    $("cxSlider").max = Math.max(0, (m.ortho.sagittal.n || 1) - 1);
    var iAx = Math.max(0, (meta.z0_index != null && meta.z_i0 != null) ? (meta.z0_index - meta.z_i0) : ((nAx / 2) | 0));
    var iCy = m.ortho.coronal.mid != null ? Math.max(0, m.ortho.coronal.mid - m.ortho.coronal.i0) : ((m.ortho.coronal.n / 2) | 0);
    var iCx = m.ortho.sagittal.mid != null ? Math.max(0, m.ortho.sagittal.mid - m.ortho.sagittal.i0) : ((m.ortho.sagittal.n / 2) | 0);
    painters.axial = RA.Planes.create($("view"), m, "axial", { i: iAx });
    painters.coronal = RA.Planes.create($("cor"), m, "coronal", { i: iCy });
    painters.sagittal = RA.Planes.create($("sag"), m, "sagittal", { i: iCx });
    sizeCanvas($("view"), m, "axial");
    sizeCanvas($("cor"), m, "coronal");
    sizeCanvas($("sag"), m, "sagittal");
    painters.axial.attachProbe($("probeAxial"));
    painters.coronal.attachProbe($("probeCor"));
    painters.sagittal.attachProbe($("probeSag"));
    [["zSlider", "zRead", painters.axial, iAx, nAx],
     ["cySlider", "cyRead", painters.coronal, iCy, m.ortho.coronal.n],
     ["cxSlider", "cxRead", painters.sagittal, iCx, m.ortho.sagittal.n]].forEach(function (tr) {
      $(tr[0]).value = tr[3];
      $(tr[1]).textContent = (tr[3] + 1) + "/" + tr[4];
    });
    prefetchAll();

    syncStates();
    redrawAll();

    /* 3D + panels */
    RA.View3D.render($("vol"), m, $("glNote"), { lam: curLam(), showLive: wants.live, vis: readVis() });
    RA.Curves.renderOperating($("curves"), m, state);
    RA.Curves.renderVol($("volcurve"), m);
    RA.Curves.renderCoverage($("coverage"), m);
    RA.Curves.renderNote($("certNote"), m);
    RA.Curves.renderContact($("contact"), m.extras.contact, m.meta);
    RA.Curves.renderCohortPng($("cohortFig"), m.extras.cohort_png);
    renderPayloadMap();
    renderExport();
    RA.View3D.render($("vol"), m, $("glNote"), { lam: curLam(), showLive: wants.live, vis: readVis() });
    updateArLink();            /* AR: point the 3D-panel link at this case */

    $("status").textContent = m.pid + " · loaded · " +
      m.report.sections.filter(function (s) { return s.status === "consumed"; }).length + "/" +
      m.report.sections.length + " payload sections consumed";
    console.log("[reading-aid] payload map for " + m.pid);
    console.table(m.report.sections);
    m.report.notes.forEach(function (nn) { console.warn("[reading-aid] " + nn); });
  }

  function curLam() {
    return model ? (model.curves ? RA.lamFromCurves(model.curves, state.alpha, model.meta.lam) : model.meta.lam) : null;
  }

  function sizeCanvas(cv, m, plane) {
    var bgk = plane === "axial" ? m.slices.bgs : ((m.ortho[plane] || {}).bgs || {});
    var first = Object.keys(bgk).length ? bgk[Object.keys(bgk)[0]] : null;
    if (first && first.length) {
      var dims = null;
      try { dims = RA.PNG.dims(first[0]); } catch (e) { }
      if (dims) {
        cv.width = dims.width; cv.height = dims.height;
        cv.style.aspectRatio = dims.width + " / " + dims.height;
        return;
      }
      var im = RA.Planes.imgOf(first[0], function () { sizeCanvas(cv, m, plane); if (painters[plane]) painters[plane].draw(); });
      if (im) {
        cv.width = im.naturalWidth || 256; cv.height = im.naturalHeight || 256;
        cv.style.aspectRatio = cv.width + " / " + cv.height;
        return;
      }
    }
    if (m.V3.ready) {
      var ax = RA.PLANE_AXES[plane];
      cv.width = m.V3.n[ax.cols] * 8; cv.height = m.V3.n[ax.rows] * 8;
      cv.style.aspectRatio = cv.width + " / " + cv.height;
    } else { cv.width = 256; cv.height = 256; }
  }

  function prefetchAll() {
    if (!model) return;
    RA.regionPrefetch(model, painters.axial.state.i, 1);
    [["axial", model.slices.bgs], ["coronal", (model.ortho.coronal || {}).bgs], ["sagittal", (model.ortho.sagittal || {}).bgs]]
      .forEach(function (pr) {
        var bgs = pr[1] || {};
        Object.keys(bgs).forEach(function (k) { RA.preload(bgs[k], painters[pr[0]].state.i, 2); });
        Object.keys(model.slices.comps || {}).forEach(function (role) {
          if (pr[0] === "axial") RA.preload(model.slices.comps[role], painters.axial.state.i, 1);
        });
      });
  }

  function syncStates() {
    Object.keys(painters).forEach(function (p) {
      var st = painters[p].state;
      st.alpha = state.alpha; st.beta = state.beta; st.win = state.win; st.lvl = state.lvl;
      Object.keys(wants).forEach(function (w) { st.wants[w] = wants[w]; });
    });
    var lam = curLam();
    $("lamRead").textContent = lam != null ? "λ* = " + Number(lam).toFixed(2) : "λ* —";
    $("alphaRead").textContent = "α ≤ " + state.alpha.toFixed(2);
    $("betaRead").textContent = "β ≤ " + state.beta.toFixed(2);
    var cert = model ? model.defaults : {};
    $("certRead").textContent = (cert.alpha != null && cert.beta != null)
      ? ("certified point α=" + cert.alpha + " β=" + cert.beta) : "";
    $("certRead").classList.toggle("at-cert", cert.alpha === state.alpha && cert.beta === state.beta);
    /* beta admissibility of the current lambda*: FP bound at jFn <= beta ? */
    if (model && model.curves && model.curves.fp_mean) {
      var gb = RA.fpBoundAt(model.curves, state.alpha);
      var chip = $("admissChip");
      if (gb != null) {
        chip.style.display = "";
        chip.textContent = "FP bound at λ*: " + gb.toFixed(3) + (gb <= state.beta ? " ≤ β ✓ admissible" : " > β ✗ NOT admissible");
        chip.className = "chip " + (gb <= state.beta ? "chip-ok" : "chip-bad");
      } else chip.style.display = "none";
    } else $("admissChip").style.display = "none";
    if (model) {
      RA.Curves.highlightCoverage($("coverage"), model);
      var liveOn = lam != null && model.meta.lam != null && Math.abs(lam - model.meta.lam) > 1e-9;
      $("liveNote").textContent = liveOn
        ? "λ* moved off the certified point — 'AI@λ live' (coarse) shows the re-thresholded outline; frozen N3 composites unchanged"
        : "";
    }
  }
  function redrawAll() {
    Object.keys(painters).forEach(function (p) { painters[p].draw(); });
    if (model) {
      RA.Curves.renderOperating($("curves"), model, state);
      RA.View3D.setLambda(curLam(), wants.live);
    }
  }

  /* ------------------------------------------------------------ controls */
  function wireControls() {
    function slider(id, fn) {
      var el = $(id);
      el.addEventListener("input", function () { fn(parseFloat(el.value)); });
    }
    slider("zSlider", function (v) {
      if (!painters.axial) return;
      painters.axial.state.i = v | 0;
      $("zRead").textContent = ((v | 0) + 1) + "/" + (($("zSlider").max | 0) + 1);
      painters.axial.draw(); prefetchAll();
    });
    slider("cySlider", function (v) {
      if (!painters.coronal) return;
      painters.coronal.state.i = v | 0;
      $("cyRead").textContent = ((v | 0) + 1) + "/" + (($("cySlider").max | 0) + 1);
      painters.coronal.draw(); prefetchAll();
    });
    slider("cxSlider", function (v) {
      if (!painters.sagittal) return;
      painters.sagittal.state.i = v | 0;
      $("cxRead").textContent = ((v | 0) + 1) + "/" + (($("cxSlider").max | 0) + 1);
      painters.sagittal.draw(); prefetchAll();
    });
    slider("alphaSlider", function (v) { state.alpha = v; syncStates(); redrawAll(); });
    slider("betaSlider", function (v) { state.beta = v; syncStates(); });
    slider("winSlider", function (v) { state.win = v; syncStates(); redrawAll(); });
    slider("lvlSlider", function (v) { state.lvl = v; syncStates(); redrawAll(); });

    [["tGt", "gt"], ["tPred", "pred"], ["tLive", "live"], ["tPost", "post"], ["tErr", "err"], ["tUnc", "unc"]]
      .forEach(function (pr) {
        var b = $(pr[0]);
        b.classList.toggle("on", wants[pr[1]]);
        b.addEventListener("click", function () {
          wants[pr[1]] = !wants[pr[1]];
          b.classList.toggle("on", wants[pr[1]]);
          syncStates();                       /* copy wants into every painter BEFORE drawing */
          redrawAll();                        /* (redrawAll also feeds View3D the live-lambda state) */
        });
      });
    $("resetWL").addEventListener("click", function () {
      state.win = 1; state.lvl = 1; $("winSlider").value = 1; $("lvlSlider").value = 1;
      syncStates(); redrawAll();
    });
    $("resetOp").addEventListener("click", function () {
      if (!model) return;
      state.alpha = model.alpha0; state.beta = model.beta0;
      $("alphaSlider").value = state.alpha; $("betaSlider").value = state.beta;
      syncStates(); redrawAll();
    });

    /* 3D layer checkboxes + focus mode (v4 cockpit) */
    ["cShell", "cCut", "cGt", "cPost", "cPred", "cN3"].forEach(function (id) {
      var el = $(id);
      if (el) el.addEventListener("change", pushVis);
    });
    function zen() {
      document.body.classList.toggle("zen");
      setTimeout(function () { RA.View3D.resize(); }, 260);
    }
    if ($("zenBtn")) $("zenBtn").addEventListener("click", zen);
    if ($("zenBtn2")) $("zenBtn2").addEventListener("click", zen);
  }

  /* what the 3D stage shows: layer checkboxes (live surface follows the AI@λ toggle) */
  function readVis() {
    function chk(id, dflt) { var el = $(id); return el ? !!el.checked : dflt; }
    return { shell: chk("cShell", true), cut: chk("cCut", true), gt: chk("cGt", true),
             post: chk("cPost", true), pred: chk("cPred", false), n3: chk("cN3", false) };
  }
  function pushVis() {
    RA.View3D.setVisible(readVis());
    updateArLink();
  }
  /* ------------------------------------------------------------ AR link (3D panel)
 * One link, created on first use and parked just above the 3D caption (#glNote),
 * so no index.html change is needed.  It follows the selected case (bootCase)
 * and the layer checkboxes (pushVis): phone shows what the stage shows. */
function arSetFromVis(vis) {
  var on = [];
  if (vis.gt) on.push("gt");
  if (vis.post) on.push("post");
  if (vis.pred) on.push("pred");
  if (on.length > 1) return "all";        /* several sets ticked -> combined GLB */
  return on.length ? on[0] : "post";      /* nothing ticked -> delivered mask */
}

  function updateArLink() {
    if (!model) return;
    var link = $("arLink");
    if (!link) {
      link = document.createElement("a");
      link.id = "arLink";
      link.className = "tgl";                       /* same pill style as the toggles */
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "📱 View in AR";
      var note = $("glNote");
      if (note && note.parentNode) note.parentNode.insertBefore(link, note);
      else if ($("vol") && $("vol").parentNode) $("vol").parentNode.appendChild(link);
    }
    var set = arSetFromVis(readVis());
    link.href = "AR/" + encodeURIComponent(model.pid) + "?set=" + set;
    link.title = "Open " + model.pid + " (" + set + " layers) in browser AR — on a phone, tap \u201cView in your space\u201d";
  }

  function wireGlobal() {
    document.addEventListener("keydown", function (ev) {
      if (!model || ev.target.tagName === "INPUT") return;
      var k = ev.key;
      if (k === "ArrowRight" || k === "ArrowLeft") {
        var d2 = k === "ArrowRight" ? 1 : -1;
        var s = $("zSlider"); s.value = Math.max(0, Math.min(s.max | 0, (s.value | 0) + d2));
        s.dispatchEvent(new Event("input")); ev.preventDefault();
      } else if (k === "u" || k === "U") { $("tUnc").click(); }
      else if (k === "e" || k === "E") { $("tErr").click(); }
      else if (k === "g" || k === "G") { $("tGt").click(); }
      else if (k === "l" || k === "L") { $("tLive").click(); }
    });

    $("mapBtn").addEventListener("click", function () { $("drawer").classList.toggle("open"); });
    $("drawerClose").addEventListener("click", function () { $("drawer").classList.remove("open"); });
  }

  /* ------------------------------------------------------------ payload map drawer */
  function renderPayloadMap() {
    var d = $("mapBody");
    var badge = { consumed: "chip-ok", partial: "chip-warn", unused: "chip-warn", absent: "chip-bad" };
    d.innerHTML = "<table class='kv'>" + model.report.sections.map(function (s) {
      return "<tr><td><b>" + s.name + "</b> <span class='chip " + (badge[s.status] || "") + "'>" + s.status + "</span></td>" +
             "<td class='detail'>" + s.detail + "</td></tr>";
    }).join("") + "</table>" +
      "<h4>coarse volume layout</h4><div class='hint'>" + (model.V3.layout || "—") +
      " · idx(x,y,z) = (x·n₁+y)·n₂+z · n=" + model.V3.n.join("×") + " · full=" + model.V3.full.join("×") + "</div>" +
      (model.report.notes.length ? "<div class='notes'>" + model.report.notes.map(function (n) { return "<div>· " + n + "</div>"; }).join("") + "</div>" : "");
  }

  /* ------------------------------------------------------------ .nii export */
  function niiBytes(vol) {
    var X = vol.niiDims[0], Y = vol.niiDims[1], Z = vol.niiDims[2];
    var hdr = new ArrayBuffer(352), dv = new DataView(hdr);
    dv.setInt32(0, 348, true);
    dv.setInt16(40, 3, true); dv.setInt16(42, X, true); dv.setInt16(44, Y, true); dv.setInt16(46, Z, true);
    dv.setInt16(48, 1, true);
    dv.setInt16(70, 2, true);                 /* datatype uint8 */
    dv.setInt16(72, 8, true);                 /* bitpix */
    var mm = (model.meta.export && model.meta.export.voxel_mm) || [1, 1, 1];
    dv.setFloat32(76, 1, true);
    dv.setFloat32(80, mm[2] || 1, true);      /* NIfTI dims are (X,Y,Z); export shape was (Z,Y,X) */
    dv.setFloat32(84, mm[1] || 1, true);
    dv.setFloat32(88, mm[0] || 1, true);
    dv.setFloat32(108, 352, true);            /* vox_offset */
    dv.setFloat32(112, 1, true);              /* scl_slope */
    new Uint8Array(hdr, 344, 4).set([0x6e, 0x2b, 0x31, 0x00]);   /* "n+1\0" */
    var out = new Uint8Array(352 + vol.data.length);
    out.set(new Uint8Array(hdr), 0);
    out.set(vol.data, 352);
    return out;
  }
  function renderExport() {
    var b = $("niiBtn");
    var vol = RA.decodeExport(model.meta || {});
    if (!vol) {
      b.disabled = true;
      b.title = "meta.export.rle absent or unrecognised -- no exportable mask in this payload";
      b.textContent = ".nii export —";
      b.onclick = null;
      return;
    }
    b.disabled = false;
    b.title = "delivered mask, " + vol.shape.join("×") + " (Z,Y,X) uint8, " + vol.count.toLocaleString() +
      " voxels, " + vol.nRuns.toLocaleString() + " runs -- decoded from meta.export.rle";
    b.textContent = ".nii export ↓";
    b.onclick = function () {
      var bytes = niiBytes(vol);
      var blob = new Blob([bytes], { type: "application/octet-stream" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = model.pid + "_post.nii";
      a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
    };
  }

  /* ------------------------------------------------------------ boot */
  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", function () {
      wireControls();
      wireGlobal();
      loadWorklist();
      $("appTag").textContent = "app v4 · cockpit layout + rebuilt head shell";
    });
  }

  RA.app = { selectCase: selectCase, state: state, wants: wants, get model() { return model; },
             get painters() { return painters; }, niiBytes: niiBytes };
})(typeof window !== "undefined" ? window : globalThis);
