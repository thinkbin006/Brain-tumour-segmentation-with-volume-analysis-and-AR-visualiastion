# Brain Tumour Volume with a Tested Error Bar

**Automatic glioma segmentation on MRI, with a volume range whose error promises were verified on unseen scans — plus a browser/AR reading aid.**

> **Research prototype — not a medical device.** Built on a public research dataset (BraTS 2020). It has not been validated on patients and must **not** be used for diagnosis or treatment decisions.

---

## What this project does

Tumour volume guides treatment, but hand-tracing a tumour across ~155 MRI slices is slow and varies between readers — and the resulting number arrives without any statement of how wrong it might be.

This project builds a pipeline that:

1. **Segments** a glioma from four MRI sequences (FLAIR, T1, T1ce, T2) with a 3D SegResNet, producing the standard nested regions — whole tumour (WT), tumour core (TC), enhancing tumour (ET).
2. **Chooses its decision cut-off statistically**, not by habit: *conformal risk control* picks λ\* so that two risks stay inside pre-set budgets — at most **10%** of true tumour missed and at most **20%** of the outline being false tissue.
3. **Reports a certified volume range.** From those budgets come the bounds `true volume ∈ [ (1−β)·v , v/(1−α) ]` — for example, a measured 52 cm³ implies roughly 42–58 cm³.
4. **Verifies the promise on sealed scans**, and flags the cases where it cannot be certified (low-contrast "faint" scans) instead of quietly reporting them as if they were.
5. **Presents everything in a reading aid** — slice viewer, overlay layers, α/β sliders, uncertainty voxels, and a 3D view that opens in a browser and in **WebXR AR on an Android phone**.

The novelty is not the network. It is the **certification layer**: a per-scan statement of how far the volume can be trusted, with the claim measured on data the pipeline never saw.

---

## Results (sealed test set — 37 scans, touched once after freeze)

| Metric | Result |
|---|---|
| Whole-tumour Dice (median) | **0.928** (TC 0.916 · ET 0.871) |
| HD95, whole tumour (median) | **≈ 3.0 mm** |
| Volume error (median) | **6.8%** (≈ 7 cm³ on a 100 cm³ tumour) |
| Scans within 20% volume error | **35 / 37** |
| Promised ≤ 10% missed · ≤ 20% wrong | **7.6% missed · 10.1% wrong — both held** |
| Certified range contained the expert volume | **35 / 37 (95%)** · CI 82–99% |
| Lesions ≥ 1 cm³ detected | **39 / 39** (they hold 99.9% of tumour volume) |
| Faint scans (flagged, not certified) | 11.8% missed vs 4.4% on clear scans |

**Honest framing:** these promises hold *on average over scans*, not for every single scan; the range is derived algebraically from two controlled averages, **not** a per-case confidence interval. With n = 37 no claim is made about beating published leaderboards. See [Limitations](#limitations).

---

## Repository layout

```
.
├── app/
│   ├── server.py               # the reading-aid server (Python stdlib only)
│   ├── ar_glb.py               # stdlib mesh3d → binary glTF (.glb) converter, v3
│   ├── static/                 # index page and viewer assets
│   └── data/
│       └── BraTS20_Training_001.json   # demo study: slices, curves, 10 mesh3d layers
├── AR_System_Doc.md            # AR viewer: endpoints, GLB spec, deployment, limits
├── docs/                       # (optional) screenshots, project guide
└── README.md
```

<!-- EDIT: add training/, notebooks/, checkpoints/ rows above if you ship that code in the same repo -->

The server computes its `static/` and `data/` paths from its own location, so the folder can be moved or cloned anywhere.

---

## Quick start

**Requirements:** Python **3.9+**. The app uses only the standard library — no `pip install` needed.

```bash
python app/server.py --port 8000 --host 0.0.0.0 --data app/data
```

Open <http://localhost:8000> → the worklist → a study.

| Option | Default | Meaning |
|---|---|---|
| `--port` | `8000` | TCP port |
| `--host` | `0.0.0.0` | listen address |
| `--data` | `app/data` | folder containing `<study>.json` payloads |

On Windows: `python app\server.py --port 8000`.

---

## The reading aid

- **Worklist** with a one-line summary per study.
- **Slice viewer** (multi-planar) for FLAIR / T1ce with window/level and a cursor probe that reports the voxel's coordinates and the model's probability there.
- **Overlay layers:** expert outline, the delivered outline, the raw model at 0.5, error layers (extra / missed), and *unsure* voxels — those whose probability sits near the cut-off — highlighted for review.
- **α / β sliders** that re-derive λ\* live and state plainly when both promises cannot hold at once.
- **3D view** of the nested surfaces, and **AR** — see below.
- **Export** of the outline as a standard `.nii` volume for other medical software.

---

## AR viewing

`/AR/<pid>` shows the study's 3D surfaces in a browser. On an Android phone in Chrome over HTTPS it offers a single button, **View in your space**, which opens an **immersive WebXR session**: point at the floor, tap to place, pinch to resize.

- Built **on demand** from the study's `mesh3d` payload — no pre-exported assets per study.
- glTF output: Y-up, metres, largest dimension **0.20 m** (display size, not calibrated), brain shell semi-transparent, tumour layers opaque.
- Desktop extras: drag to rotate, tap a point to read its layer and voxel index, export a PNG of the 3D view.
- **Android + Chrome only** for AR (WebXR); desktop and iOS get the 3D view.
- A visual aid — smoothed surfaces, not a measuring instrument.

To reach a phone, the page must be served over **HTTPS**, because WebXR requires a secure context:

```bash
# terminal 1 — the app (plain HTTP is correct)
python app/server.py --port 8000

# terminal 2 — a public HTTPS address
cloudflared tunnel --url http://localhost:8000
```

Open the printed `https://<words>.trycloudflare.com/AR/<pid>` in Chrome on the phone. The address changes on every tunnel restart. Full details: [`AR_System_Doc.md`](AR_System_Doc.md).

> There is no authentication: anyone holding the tunnel URL can open the app. Treat the link as the only access control and keep the server on a local machine.

---

## How the certificate works

1. Run the model on the **37 tuning scans** at every cut-off λ ∈ [0.05, 0.95].
2. At each λ, average the **missed share** (true tumour left out) and the **wrong share** (outline that is not tumour), using a worst-case envelope across λ.
3. Apply the finite-sample correction `n/(n+1)·mean + 1/(n+1)` — with few scans an uncorrected average would be over-optimistic.
4. Take the tightest λ whose corrected risks fit α = 10%, β = 20% → **λ\* = 0.40**.

Faint scans (lowest third of contrast-to-noise, CNR ≤ 1.49) are calibrated separately: only 12 exist in the tuning set, too few to certify, so their range is displayed but labelled **"not certified for this scan."**

The data split is fixed with a seed and never mixed: **295 train · 37 tune/calibrate · 37 sealed test** (BraTS 2020, 369 scans; ET present in 342).

---

## Model at a glance

| | |
|---|---|
| Architecture | MONAI **SegResNet**, 16 start filters, dropout 0.2, trained from scratch |
| Input / output | 4 MRI channels → 3 nested region maps (WT, TC, ET) |
| Training | 96³ patches, 70% tumour-centred, batch 2, AdamW 2×10⁻⁴, ~21.8k steps, mixed precision, ~90 min on one T4 |
| Loss | Dice + cross-entropy weighted up to **6×** near the tumour border |
| Inference | Sliding windows, **4-view flip averaging** (TTA), ~5 s per scan |
| Region cut-offs | WT searched 0.05–0.95; TC 0.45; ET 0.55; ET fragments < 400 voxels relabelled as core |

---

## Data format

Each study is a single JSON payload in `app/data/`:

| Key | Contents |
|---|---|
| `slices`, `ortho` | per-slice image and mask data for the viewers |
| `curves`, `vol_curve`, `defaults` | α/β slider curves and cut-off defaults |
| `vol3d` | full-volume dimensions for the tap readout |
| `mesh3d` | `{ n, cells, layers }` — 10 layers (`brain`, `post_wt/tc/et`, `gt_wt/tc/et`, `pred_wt/tc/et`), each base64 uint16 vertices + faces with a scale factor |
| `meta`, `coverage`, `contact` | study metadata and certification status |

`ar_glb.py` converts `mesh3d` to `.glb` on first request and caches it beside the payload with a `.sig` sidecar (payload mtime, size, converter `FORMAT_VERSION`). Delete the `*.glb` / `*.sig` files to force a rebuild.

---

## Limitations

- **Not a medical device.** No patient data, no clinical validation, no regulatory clearance.
- **Small test set.** 37 sealed scans: results are medians with confidence intervals, not leaderboard claims. ET is absent in 27 of 369 scans.
- **One dataset.** BraTS 2020 only — scanner and site shift are untested, and the certificate would need recalibration per deployment.
- **The promise is an average over scans**, not a guarantee for each individual scan. Faint scans are explicitly excluded from certification.
- **No certified blob filter.** Small false fragments are delivered unfiltered for the reader to adjudicate; filtering could not be certified at this sample size.
- **AR is presentation only** — smoothed surfaces, nominal 20 cm size, no patient coordinates. It does not affect any measurement.
- The tunnel deployment has no authentication and is intended for demos only.

---

## Acknowledgements

- **BraTS 2020** — Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark*, IEEE TMI 2015; Bakas et al., *Scientific Data* 2017.
- **MONAI** for SegResNet and medical-imaging utilities.
- Conformal risk control — Angelopoulos et al., *ICLR 2024*.

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE). The BraTS data itself is subject to its own terms and is not redistributed beyond the single demonstration payload included to run the viewer.
