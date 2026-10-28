> **Start with MERGE_NOTES.md** (what was merged from where, and a correction to my earlier kit).

# Glioma reading-aid — app package v4

The local workstation that renders **N3 notebook payloads** (`report_<pid>.json`), verified
field-by-field against a real payload (`BraTS20_Training_095`).  Offline: plotly is vendored,
there is no CDN, no build step, no dependencies beyond Python 3 (server + installer) and a
browser.

## Run it

```bash
python3 app/server.py --port 8000            # serves the workstation on http://localhost:8000
```

Install cases (files, a folder, or the Kaggle output zip all work):

```bash
python3 code/install_app_study.py ~/Downloads/report_BraTS20_Training_095.json
python3 code/install_app_study.py E:\reports            # finds report_*.json (one level deep too)
python3 code/install_app_study.py kaggle-output.zip     # reads payloads straight out of the zip
python3 code/install_app_study.py --list                # show the worklist manifest
```

Optional synthetic demo case (exercises every payload section; regenerates `app/data/synthetic-demo.json`):

```bash
python3 code/make_demo_case.py
```

## What the app shows (all of it from the payload)

| Panel | Payload source |
|---|---|
| axial / coronal / sagittal planes | `slices.flair`+`t1ce` (64 PNGs, 161×184, row=y col=x, z=z_i0+i), `ortho.*` (28 PNGs, i0 windows) |
| ref / AI@λ* / delivered / err overlays (axial) | N3's own composites `slices.gt/pred/post/err` drawn verbatim (err: red=EXTRA, blue=MISSED) |
| AI@λ live + unsure dither + ortho overlays | `vol3d.planes.*` PNG-packed coarse volumes (w=NZ, h=NX·NY → (X,Y,Z), idx=(x·n₁+y)·n₂+z) |
| hover probe | reference/delivered FULL-RES from `slices.gtw/gtc/gte` + `pw/pc/pe` (alpha>0; verified counts = meta volumes); p, λ, unsure band from coarse `wt` (the payload's only probabilities) |
| 3D view (plotly mesh3d) | head shell **rebuilt in-app** as a visual hull of the full-res background silhouettes (N3's own brain layer is an intensity-iso ribbon: 17 patches — kept as an opt-in comparison); tumour shells from `mesh3d.layers.*` **uint16 LE vertices / scale**, cut away camera-side so nested rings read; pred hidden by default where vertex-identical to post |
| 3D live AI surface | merged-vertex cuberille re-mesh of coarse `wt ≥ λ*` — follows the α slider |
| operating curves, λ* | `curves` with the certified Bayes bound (n/(n+1))·mean + 1/(n+1); λ* = last threshold with FN bound ≤ α |
| β admissibility chip | FP bound at λ* vs β |
| coverage grid | `meta.coverage_rows` (N3's own `<tr data-k="α_β">` rows) — the matching row highlights as you drag α/β |
| findings / regions / engine | `meta.ai` (every field, every region) + `meta.region_rows` + `meta.engine_rows` verbatim |
| certificate note | `meta.note`, `cert_scope`, `n_cal`, `filter_feasible/delta`, geometry (x0/y0/z0/z_full/…) |
| volume-vs-λ curve | `vol_curve {thr, v}` |
| cohort figure + contact sheet | `cohort_png`, `contact` (37 cards) |
| AI review | `meta.ai_review` (FIX v2 payloads); older payloads get an honest "predates FIX v2" notice |
| .nii export | `meta.export.rle`: int32 LE (start,len) runs over uint8 (Z,Y,X) → NIfTI-1 dims (X,Y,Z); verified count-exact (108 032 voxels) and slice-exact vs `slices.post` |
| **Payload map** button | every section classified consumed/partial/unused/absent — if N3 ever emits something the app doesn't render, it says so on screen and in the console |

Fix behaviours are native (not overrides): W/L filters the scan only (never a mask), the unsure
dither paints on every plane at the live λ, and the probe reports reference vs AI-at-λ vs
delivered vs unsure purview on all three planes.

## v4 cockpit layout & 3D

The workstation is a **single viewport**: worklist · controls · slices/numbers · 3D stage sit side
by side; only the slice column scrolls internally, so every slider/toggle/checkbox change is
visible in the planes *and* the 3D at the same time — no scrolling up and down.  `⛶ focus 3D`
folds the slice column away and gives the 3D stage the whole width.

Why the old 3D looked patchy (diagnosed in `fixes/diag_brain.js`, headless-chrome shots in
`fixes/shot_*.png`):

1. N3's `mesh3d.brain` is an isosurface of the coarse intensity map, which is a tissue *ribbon*
   (88 % of the coarse grid is zero even inside the head; the backgrounds sit on a flat per-slice
   grey floor, not black) → the shell arrives as 17 disconnected patches.
2. `pred_wt/pred_tc` are vertex-identical to `post_wt/post_tc`, and gt/post/pred nest inside each
   other; stacked translucent shells multiply alphas (≈2 % transmission) and coincident surfaces
   z-fight → plotly showed one orange blob.

v4 therefore rebuilds a whole head shell in-app (per-slice `|v−median|>T` silhouettes of the 64
axial + 28 coronal + 28 sagittal full-res backgrounds → visual-hull intersection on the coarse
grid → hole-fill + closing → half-res cuberille with merged vertices → Taubin smoothing) and
renders the tumour families as a **cutaway onion** (camera-side hemisphere clipped, innermost
surface first).  No payload change and no bigger file was needed — everything comes from stacks
the payload already carries.  Checkboxes: head shell, cutaway, gt, post, pred (off: ≡ post here),
N3 brain mesh (off: patchy comparison).

## Verified how

```bash
node fixes/test_real_payload.js uploads/report_BraTS20_Training_095.json   # 56 checks: byte-identical
                                                     # coarse decode vs python, λ*, probes (axial+coronal),
                                                     # full-res probe, mesh u16 bboxes, containment, RLE
node fixes/test_app_ui.js   uploads/report_BraTS20_Training_095.json       # 25 checks: whole app driven
                                                     # through a stub DOM — canvases, W/L isolation, every
                                                     # toggle, α/β→3D live surface, coverage highlight,
                                                     # probe readouts, .nii blob size
node fixes/test_app_adapter.js                                             # demo payload, simple schema
python3 fixes/verify_real_payload.py uploads/report_BraTS20_Training_095.json   # re-derives ground truth
```

Schema derivation + all resolved conventions: `fixes/real_schema.md`.

## Kaggle side (N3 notebook)

`fixes/N3_fix_cell.py` — paste as one cell **between the payload/`render_case` cell and the
report-writing loop** (i.e. after the `REPORT_TEMPLATE` cell and after the cell that defines
`render_case`, before the cell that calls it in a loop and writes `report_<pid>.html/.json`).
"After the template cell" alone is not enough if your template cell sits above the
`render_case` cell, and putting the fix cell at the end of the notebook is silently useless
(reports were already written). The cell starts with a placement guard that raises on a
too-early slot and warns loudly on a too-late one — `fixes/test_cell_order.py` tests all four
placements.
FIX v2.1: mask-based head shell + containment check, W/L-isolated overlays, unsure dither on all
planes, cursor probe, AI review line — **plus** an idempotent uint16 mesh re-decode for the
standalone report page (same wrap bug the app-side decode avoids).  v2.1 pages now wear a visible
badge ("page fix v2.1 · uint16 mesh decode · mask-based head shell") and expose
`window.__pageFix.version`, so you can always see which fix version a report runs; the cell prints
a `FIX v2.1 ready:` banner on install.  The real payload you shared predates the fix cell: no
`meta.ai_review` and an intensity-iso brain layer — re-run N3 with the cell to get both.

## Layout

```
app/server.py                 endpoints: /, /api/studies, /api/study/<pid>, /report/<pid>, /static/*
app/data/                     installed payloads + index.json (worklist manifest)
app/static/index.html         workstation UI
app/static/css/app.css        styling
app/static/js/png.js          dependency-free PNG decoder (browser + node)
app/static/js/payload.js      N3 adapter: schema branches, probe/λ*/containment maths
app/static/js/planes.js       2D painter (W/L isolation, overlays, dither, probe wiring)
app/static/js/view3d.js       plotly 3D v4: rebuilt head shell, cutaway onion, live-λ cuberille
app/static/js/curves.js       operating curves, vol_curve, coverage, note/rows, contact
app/static/js/app.js          wiring, panels, export, payload map
app/static/vendor/plotly.min.js   plotly 2.35.2 (MIT, vendored for offline use)
code/install_app_study.py     payload installer + contract checker
code/make_demo_case.py        synthetic demo payload + ground-truth expectations
fixes/                        N3 fix cell, page/payload fixes, verification + test suites
```

Nothing here is a model or a metric: the notebook stays the producer, the app renders its output
and labels coarse visual aids as such — every NUMBER is full-resolution.
