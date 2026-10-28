/* curves.js -- 2D plotly panels + generic tables (v3, real-payload schema).
 *   operating curves : Bayes-bounded FN/FP vs threshold with the live operating point
 *   vol_curve        : {thr:[], v:[]} raw AI volume (mm3) vs threshold
 *   coverage         : list of certified (alpha,beta) rows -> table with live highlight;
 *                      if meta.coverage_rows (pre-rendered <tr data-k="a_b">) exists it is
 *                      used verbatim so the numbers shown are exactly N3's
 *   contact          : cohort cards {id,tag,dice,miss,img}
 *   cohort_png       : single cohort figure
 */
(function (root) {
  "use strict";
  var RA = root.RA || (root.RA = {});

  var FONT = { family: "ui-monospace, SFMono-Regular, Menlo, monospace", size: 10, color: "#9fb0c0" };
  function axes(title) {
    return { gridcolor: "rgba(140,160,180,0.12)", zeroline: false, linecolor: "rgba(140,160,180,0.25)",
             tickfont: FONT, title: title ? { text: title, font: FONT } : undefined };
  }
  function noPlotly(div, msg) { div.innerHTML = "<div class='empty3d'>" + msg + "</div>"; }

  /* ---- operating curves ---- */
  function renderOperating(div, model, state) {
    if (typeof root.Plotly === "undefined") return noPlotly(div, "plotly did not load");
    if (!model.curves) return noPlotly(div, "no curves section in this payload");
    var c = model.curves, n = c.n_cal || 1, t = c.thresholds;
    var bnd = function (arr) { return arr.map(function (m) { return (n / (n + 1)) * m + 1 / (n + 1); }); };
    var fb = bnd(c.fn_mean), gb = c.fp_mean ? bnd(c.fp_mean) : null;
    var lam = RA.lamFromCurves(c, state.alpha, model.meta.lam);
    var data = [
      { x: t, y: fb, name: "FN bound (α)", mode: "lines+markers",
        line: { color: "#ff6b6b", width: 1.6 }, marker: { size: 3 } }
    ];
    if (gb) data.push({ x: t, y: gb, name: "FP bound (β)", mode: "lines+markers",
      line: { color: "#4dabf7", width: 1.6 }, marker: { size: 3 } });
    data.push({ x: [lam], y: [0], mode: "markers", name: "λ* = " + Number(lam).toFixed(2),
      marker: { color: "#ffa94d", size: 9, symbol: "triangle-up" } });
    var layout = {
      paper_bgcolor: "rgba(0,0,0,0)", margin: { l: 34, r: 8, t: 24, b: 28 },
      font: FONT, showlegend: true,
      legend: { orientation: "h", y: 1.22, x: 0, font: FONT, bgcolor: "rgba(0,0,0,0)" },
      xaxis: axes("threshold λ"), yaxis: axes("bounded rate"),
      shapes: [
        { type: "line", x0: lam, x1: lam, y0: 0, y1: 1, yref: "paper",
          line: { color: "rgba(255,169,77,0.55)", width: 1, dash: "dot" } },
        { type: "line", x0: 0, x1: 1, xref: "paper", y0: state.alpha, y1: state.alpha,
          line: { color: "rgba(255,107,107,0.35)", width: 1, dash: "dash" } }
      ],
      annotations: [{ x: 1, xref: "paper", y: state.alpha, text: "α=" + state.alpha.toFixed(2),
        showarrow: false, font: { size: 9, color: "#ff8787" }, xanchor: "right", yshift: 8 }]
    };
    if (gb) {
      layout.shapes.push({ type: "line", x0: 0, x1: 1, xref: "paper", y0: state.beta, y1: state.beta,
        line: { color: "rgba(77,171,247,0.35)", width: 1, dash: "dash" } });
      layout.annotations.push({ x: 1, xref: "paper", y: state.beta, text: "β=" + state.beta.toFixed(2),
        showarrow: false, font: { size: 9, color: "#74c0fc" }, xanchor: "right", yshift: -10 });
    }
    root.Plotly.react(div, data, layout, { responsive: true, displayModeBar: false });
  }

  /* ---- vol_curve: {thr:[], v:[]} (mm3) or generic numeric-arrays ---- */
  function renderVol(div, model) {
    if (typeof root.Plotly === "undefined") return noPlotly(div, "plotly did not load");
    var vc = model.extras.vol_curve;
    if (!vc) return noPlotly(div, "no vol_curve section");
    var xs = vc.thr || vc.lam || vc.thresholds || vc.x;
    var ys = vc.v || vc.vol || vc.volume;
    if (!Array.isArray(xs)) {
      var keys = Object.keys(vc).filter(function (k) { return Array.isArray(vc[k]) && typeof vc[k][0] === "number"; });
      if (keys.length >= 2) { xs = vc[keys[0]]; ys = vc[keys[1]]; }
    }
    if (!Array.isArray(xs) || !Array.isArray(ys)) return noPlotly(div, "vol_curve: no numeric thr/v arrays");
    var lam = RA.lamFromCurves(model.curves, model.__state ? model.__state.alpha : model.alpha0, model.meta.lam);
    root.Plotly.react(div, [
      { x: xs, y: ys.map(function (v) { return v / 1000; }), name: "raw AI volume",
        mode: "lines+markers", line: { color: "#ffa94d", width: 1.6 }, marker: { size: 3 } },
      { x: [lam], y: [0], mode: "markers", name: "λ* " + Number(lam).toFixed(2),
        marker: { color: "#ffd8a8", size: 9, symbol: "triangle-up" } }
    ], {
      paper_bgcolor: "rgba(0,0,0,0)", margin: { l: 40, r: 8, t: 22, b: 28 }, font: FONT,
      showlegend: true, legend: { orientation: "h", y: 1.22, x: 0, font: FONT, bgcolor: "rgba(0,0,0,0)" },
      xaxis: axes("threshold λ"), yaxis: axes("volume cm³")
    }, { responsive: true, displayModeBar: false });
  }

  /* ---- coverage: prefer N3's pre-rendered rows (exact numbers), fall back to the list ---- */
  function renderCoverage(div, model) {
    var rowsHtml = model.meta.coverage_rows;
    var head = "<tr><td>α</td><td>β</td><td>λ</td><td>missed</td><td>wrong</td><td>contain.</td></tr>";
    if (typeof rowsHtml === "string" && rowsHtml.indexOf("<tr") >= 0) {
      div.innerHTML = "<table class='kv covtable'>" + head + rowsHtml + "</table>";
    } else if (Array.isArray(model.extras.coverage) && model.extras.coverage.length) {
      div.innerHTML = "<table class='kv covtable'>" + head + model.extras.coverage.map(function (r) {
        return "<tr data-k='" + Number(r.alpha).toFixed(2) + "_" + Number(r.beta).toFixed(2) + "'>" +
          "<td>" + r.alpha + "</td><td>" + r.beta + "</td><td>" + r.lam + "</td>" +
          "<td>" + RA.fmt.num(r.realised_missed, 3) + "</td><td>" + RA.fmt.num(r.realised_wrong, 3) + "</td>" +
          "<td>" + RA.fmt.pct(r.containment, 0) + "</td></tr>";
      }).join("") + "</table>";
    } else {
      return noPlotly(div, "no coverage section");
    }
    highlightCoverage(div, model);
  }
  function highlightCoverage(div, model) {
    var st = model.__state;
    if (!st) return;
    var key = Number(st.alpha).toFixed(2) + "_" + Number(st.beta).toFixed(2);
    var rows = div.querySelectorAll ? div.querySelectorAll("tr[data-k]") : [];
    for (var i = 0; i < rows.length; i++) rows[i].classList.toggle("hot", rows[i].getAttribute("data-k") === key);
  }

  /* ---- meta note + region/engine tables (N3's own pre-rendered HTML rows) ---- */
  function renderNote(div, model) {
    var bits = [];
    if (model.meta.note) bits.push("<p class='note'>" + model.meta.note + "</p>");
    if (model.meta.region_rows) {
      bits.push("<h4>regions (full-resolution counts)</h4><table class='kv'>" +
        "<tr><td>region</td><td>GT mm³</td><td>delivered mm³</td><td>Δ</td><td>Dice</td></tr>" +
        model.meta.region_rows + "</table>");
    }
    if (model.meta.engine_rows) {
      bits.push("<h4>engine vs delivered</h4><table class='kv'>" + model.meta.engine_rows + "</table>");
    }
    var geo = ["x0", "y0", "z0", "z0_index", "z_full", "z_i0", "n_axial", "filter_feasible", "filter_delta"]
      .filter(function (k) { return model.meta[k] != null; });
    if (geo.length) {
      bits.push("<h4>geometry / post-processing</h4><table class='kv'>" + geo.map(function (k) {
        return "<tr><td>" + RA.fmt.pretty(k) + "</td><td class='num'>" + model.meta[k] + "</td></tr>";
      }).join("") + "</table>");
    }
    div.innerHTML = bits.join("") || "<div class='empty3d'>no note/rows in meta</div>";
  }

  /* ---- contact sheet: cohort cards ---- */
  function renderContact(div, contact, meta) {
    /* meta.lesion_min_vox is written by N3 (final): `miss` then means "a lesion of at least that many voxels was missed".
       Older payloads flagged ANY missed component, which is mostly annotation specks, so say so instead of "lesion". */
    var minVox = meta && meta.lesion_min_vox;
    var missText = function (c) {
      if (c.miss == null) return "";
      if (minVox) return c.miss ? " · <span class='bad'>lesion ≥ " + (minVox / 1000).toFixed(minVox % 1000 ? 1 : 0) + " cm³ missed</span>"
                                : " · all lesions ≥ " + (minVox / 1000).toFixed(minVox % 1000 ? 1 : 0) + " cm³ found";
      return " · " + (c.miss ? "a fragment of any size missed (older payload: mostly specks)" : "no fragment missed");
    };
    if (!Array.isArray(contact) || !contact.length) return noPlotly(div, "no contact section");
    div.innerHTML = "<div class='contactgrid'>" + contact.map(function (c) {
      if (typeof c === "string") return "<img class='contact-img' src='" + c + "' alt='contact'>";
      return "<figure class='contact-card'>" +
        (c.img ? "<img loading='lazy' src='" + c.img + "' alt='" + (c.id || "") + "'>" : "") +
        "<figcaption><b>" + (c.id || "") + "</b><br>" + (c.tag || "") +
        (c.dice != null ? " · Dice " + Number(c.dice).toFixed(2) : "") +
        missText(c) + "</figcaption></figure>";
    }).join("") + "</div>";
  }
  function renderCohortPng(div, uri) {
    if (typeof uri !== "string" || uri.indexOf("data:") !== 0) return noPlotly(div, "no cohort_png");
    div.innerHTML = "<img class='contact-img' src='" + uri + "' alt='cohort figure'>";
  }

  RA.Curves = { renderOperating: renderOperating, renderVol: renderVol, renderCoverage: renderCoverage,
                highlightCoverage: highlightCoverage, renderNote: renderNote,
                renderContact: renderContact, renderCohortPng: renderCohortPng };
})(typeof window !== "undefined" ? window : globalThis);
