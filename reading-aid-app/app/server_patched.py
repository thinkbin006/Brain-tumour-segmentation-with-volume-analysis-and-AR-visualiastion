#!/usr/bin/env python3
"""Local reading-aid app -- server (v2 + AR patch).

Everything from v2, unchanged, plus browser-based AR (Option 1: <model-viewer>):

    /AR/<pid>               AR page for a study (3D on desktop, "View in your space" on a phone)
                            ?set=post|gt|pred|all   which mesh3d layers (default post), or
                            ?layers=brain,gt_wt,gt_tc,gt_et   exact layers (wins over ?set=;
                            omit `brain` for no shell) -- this mirrors the 3D panel checkboxes
    /api/AR/<pid>.glb       the study's mesh3d layers as binary glTF, built on first request
                            by app/ar_glb.py and cached in the data dir (rebuilt when the
                            payload's mtime/size changes).  ?set= / ?layers= as above.
                            400 on an unknown layer name, 404 if the payload has no mesh3d.
    /api/AR/<pid>.usdz      served if you drop a converted <pid>.usdz in the data dir (iOS only;
                            without it iPhone shows the interactive 3D view but no AR button)
    /api/AR/<pid>.json      coordinate metadata (layers + export transform) for the AR
                            page's tap-to-coordinates: a tapped model point is converted
                            back to mesh3d grid indices (see ar_glb.ar_metadata)
    /ARXR/<pid>             experimental WebXR AR page (Android Chrome + HTTPS only): the
                            browser keeps the AR session, so taps INSIDE AR raycast the
                            model and report coordinates + the exact layer hit.  The
                            regular /AR page (native Scene Viewer / Quick Look) cannot
                            do this -- the OS owns that session and sends no taps back.

No new dependencies: ar_glb.py is stdlib-only, and the GLB is generated from the mesh3d
section N3 already writes -- no re-segmentation, no marching cubes at serve time.

Run it exactly as before:
    python3 app/server.py [--port 8000] [--host 0.0.0.0] [--data app/data]

For AR on a real phone the page must be HTTPS (WebXR/camera rule; localhost is exempt but a
phone is not localhost). Tunnel your usual port, e.g.:   ngrok http 8000
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "app", "static")
DATA = os.environ.get("APP_DATA") or os.path.join(ROOT, "app", "data")

# AR: ar_glb.py sits next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import ar_glb
except Exception:                                    # server still runs without AR
    ar_glb = None

MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".map": "application/json; charset=utf-8",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml", ".ico": "image/x-icon", ".webp": "image/webp",
        ".nii": "application/octet-stream", ".gz": "application/gzip",
        ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf",
        # AR
        ".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
        ".usdz": "model/vnd.usdz+zip"}

# what each study is here to show -- the app labels the case list with it, so the demo has a story
NOTES = {
    "satellite":   ("missed satellite", "a second small lesion across the midline that the model does not find"),
    "lowcontrast": ("low contrast", "a faint, patchy lesion: the threshold sits inside the noise"),
    "multifocal":  ("multifocal", "two lesions the model finds but BRIDGES into one"),
    "typical":     ("single lesion", "a plain enhancing lesion with an oedematous rim"),
}

# ---------------------------------------------------------------- summary cache
_CACHE = {}
_CACHE_LOCK = threading.Lock()
_AR_LOCK = threading.Lock()          # AR: one GLB build at a time


def _manifest():
    try:
        with open(os.path.join(DATA, "index.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _summary(path):
    """Tiny per-study dict for the worklist, cached by (mtime, size).  None = not a study."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = path
    sig = (st.st_mtime_ns, st.st_size)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return None
    m = d.get("meta")
    if not isinstance(m, dict):                 # anything that is not a study payload
        return None
    ai = (m.get("ai") or {}).get("WT", {})
    out = dict(lam=m.get("lam"), gt_mm3=m.get("gt_volume"), ai_mm3=m.get("post_volume"),
               covered=ai.get("covered"), extra=ai.get("extra"), dice=ai.get("agree"),
               scope=m.get("cert_scope"), n_cal=m.get("n_cal"),
               has_ar_mesh=bool(((d.get("mesh3d") or {}).get("layers"))),
               has_review=bool((m.get("ai_review") or "").strip()))
    with _CACHE_LOCK:
        _CACHE[key] = (sig, out)
    return out


def _studies():
    out = []
    man = _manifest()
    if not os.path.isdir(DATA):
        return out
    for f in sorted(os.listdir(DATA)):
        if not f.endswith(".json") or f.startswith("report_") or f == "index.json":
            continue
        pid = f[:-5]
        s = _summary(os.path.join(DATA, f))
        if s is None:
            continue
        mv = (man.get(pid) or {})
        note = NOTES.get(mv.get("variant", ""), ("study", ""))
        tag = mv.get("tag") or note[0]
        why = mv.get("why") or note[1]
        # NOTE: _summary returns the CACHED dict -- never pop() from it, or the
        # key vanishes for every later request (that bug made has_ar true once).
        has_mesh = bool(s.get("has_ar_mesh", False))
        out.append(dict(pid=pid, tag=tag, why=why,
                        patient=mv.get("patient") or pid,
                        variant=mv.get("variant"),
                        has_report=os.path.exists(os.path.join(DATA, pid + ".html")),
                        # AR: link only when the converter is present AND the payload
                        # really has mesh3d layers (checked in _summary, not guessed here)
                        has_ar=bool(ar_glb) and has_mesh,
                        **{k: v for k, v in s.items() if k != "has_ar_mesh"}))
    return out


def _contained(directory, name):
    """Resolve `name` inside `directory`; return None if it escapes (any OS, any encoding trick)."""
    p = os.path.realpath(os.path.join(directory, name))
    root = os.path.realpath(directory)
    if p == root or p.startswith(root + os.sep):
        return p
    return None


# ---------------------------------------------------------------- AR helpers
_AR_SETS = ("post", "gt", "pred", "all")


def _ar_set(query):
    s = (parse_qs(query).get("set") or ["post"])[0].lower()
    return s if s in _AR_SETS else "post"


def _ar_layers(query):
    """Exact ?layers=brain,gt_wt,... spec, or None to fall back to ?set=.

    Whitelist-validated against ar_glb.LAYER_ORDER here (bad name -> ValueError
    -> 400, not a silent wrong model).  Normalised to canonical order so the
    cache key is stable however the UI ordered the checkboxes.
    """
    raw = (parse_qs(query).get("layers") or [None])[0]
    if not raw:
        return None
    if ar_glb is None:
        raise ValueError("AR converter unavailable")
    asked = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = asked - set(ar_glb.LAYER_ORDER)
    if unknown:
        raise ValueError("unknown layer(s): %s" % ", ".join(sorted(unknown)))
    if not asked:
        raise ValueError("empty layers")
    return ",".join(k for k in ar_glb.LAYER_ORDER if k in asked)


def _ar_glb_path(pid, layer_set, layers=None):
    """Path of the study's cached .glb, building it from the payload on first use.

    Cache rule mirrors _summary: the .glb is stale the moment the payload's
    (mtime, size) no longer matches what the .glb was built from -- tracked by
    simply rebuilding whenever payload mtime >= glb mtime is not enough after a
    same-second reinstall, so the signature is kept in a tiny sidecar file.
    """
    if ar_glb is None:
        return None
    src = _contained(DATA, pid + ".json")
    if not src or not os.path.isfile(src):
        return None
    if layers:                                   # exact spec -> hashed cache name
        import hashlib
        tag = "L" + hashlib.sha1(layers.encode()).hexdigest()[:10]
    else:
        tag = layer_set
    out = _contained(DATA, "%s.%s.glb" % (pid, tag))
    sig_path = out + ".sig"
    try:
        st = os.stat(src)
        sig = "%d:%d:%s" % (st.st_mtime_ns, st.st_size, layers or layer_set)
    except OSError:
        return None
    with _AR_LOCK:
        try:
            with open(sig_path, encoding="utf-8") as fh:
                if fh.read().strip() == sig and os.path.isfile(out):
                    return out
        except Exception:
            pass
        try:
            ar_glb.convert_file(src, out, layer_set, layers=layers)
            with open(sig_path, "w", encoding="utf-8") as fh:
                fh.write(sig)
        except Exception as e:
            print("  AR: could not build %s: %s" % (os.path.basename(out), e))
            return None
    return out


_AR_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PID__ -- AR 3D</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#0d1117;color:#e6edf3}
  model-viewer{width:100%;height:72vh;background:#161b22;border-radius:12px}
  #ar-button{background:#e5534b;color:#fff;border:0;border-radius:8px;padding:12px 18px;
             font-size:16px;position:absolute;bottom:16px;right:16px;cursor:pointer}
  .hint{color:#8b949e;font-size:14px} a{color:#58a6ff}
  .sets a{margin-right:10px}
  #coords{background:#161b22;border:1px solid #2a313c;border-radius:8px;padding:10px 12px;
          font:13px ui-monospace,monospace;white-space:pre-wrap;margin-top:10px;min-height:3.2em}
  .hotspot{width:14px;height:14px;border-radius:50%;border:2px solid #fff;background:#e5534b;
           box-shadow:0 0 6px #000;cursor:pointer}
  label.pick{font-size:14px}
</style>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
</head><body>
  <p><a href="/">&larr; worklist</a></p>
  <h2>__PID__ &middot; 3D segmentation</h2>
  <p class="hint">Drag to rotate. On a phone, tap <b>View in your space</b> and point at a
     table -- the brain is placed 20&nbsp;cm across, brain shell translucent, tumour inside.</p>
  <p class="sets">Layers:
    <a href="/AR/__PID__?set=post">delivered (post)</a>
    <a href="/AR/__PID__?set=gt">ground truth</a>
    <a href="/AR/__PID__?set=pred">raw model</a>
    <a href="/AR/__PID__?set=all">all</a>
    &middot; <a href="/api/AR/__PID__.glb?__GLBQ__">download .glb</a>
    &middot; <a href="/ARXR/__PID__?__GLBQ__"><b>Android WebXR AR with in-AR tap coordinates</b></a></p>

  <model-viewer id="mv" src="/api/AR/__PID__.glb?__GLBQ__"
                ios-src="/api/AR/__PID__.usdz"
                alt="3D brain tumour segmentation for __PID__"
                AR AR-modes="webXR scene-viewer quick-look" AR-scale="auto"
                camera-controls auto-rotate shadow-intensity="1" exposure="1.1">
    <button slot="ar-button" id="ar-button">View in your space (AR)</button>
  </model-viewer>

  <p><label class="pick"><input type="checkbox" id="pickMode"> 📍 Tap-to-coordinates
     <span class="hint">— then tap (don't drag) any point on the model in the 3D view above.
     Works in the in-browser 3D view; inside native AR (Scene Viewer / Quick Look) the
     page is handed off to the OS, so tapping points there is not available.</span></label></p>
  <div id="coords">Coordinates will appear here. Turn on tap-to-coordinates and tap the tumour.</div>
  <p><button id="copyCoords" type="button">Copy coordinates (JSON)</button>
     <button id="clearCoords" type="button">Clear points</button></p>
  <p class="hint">Crop-local coordinates, as in the .nii export -- no world orientation is
     claimed. Layers: <b>__SET__</b>. Frozen N3 meshes at the certified &lambda;* (a live-&lambda;
     re-threshold exists only as coarse volumes in the payload, so it cannot be re-meshed
     faithfully here).</p>
<script>
(function () {
  var mv = document.getElementById("mv"), box = document.getElementById("coords");
  var meta = null, last = null, n = 0, down = null;
  /* transform for THIS layer selection (inverse maths documented in ar_glb._frame) */
  fetch("/api/AR/__PID__.json?__GLBQ__").then(function (r) { return r.json(); })
    .then(function (j) { meta = j; })
    .catch(function (e) { box.textContent = "coordinate metadata unavailable: " + e; });

  mv.addEventListener("pointerdown", function (e) { down = [e.clientX, e.clientY]; });
  mv.addEventListener("pointerup", function (e) {
    if (!document.getElementById("pickMode").checked || !meta || !down) return;
    var moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]); down = null;
    if (moved > 6) return;                       /* it was a drag (orbit), not a tap */
    var rect = mv.getBoundingClientRect();
    var hit = mv.positionAndNormalFromPoint(e.clientX - rect.left, e.clientY - rect.top);
    if (!hit || !hit.position) { box.textContent = "No surface at that point — tap directly on the model."; return; }
    var p = hit.position, t = meta.transform, s = t.scale_m_per_grid_unit;
    /* glTF metres -> mesh3d grid indices (exact inverse of the export transform) */
    var grid = [p.x / s + t.grid_centre[0], -p.z / s + t.grid_centre[1], p.y / s + t.grid_lo[2]];
    var pct = meta.mesh_n ? grid.map(function (v, i) { return Math.round(1000 * v / meta.mesh_n[i]) / 10; }) : null;
    var full = (meta.mesh_n && meta.mesh_full) ? grid.map(function (v, i) {
        return Math.round(10 * v * (meta.mesh_full[i] - 1) / (meta.mesh_n[i] - 1)) / 10; }) : null;
    n += 1;
    var hs = document.createElement("button");
    hs.className = "hotspot"; hs.slot = "hotspot-" + n;
    hs.setAttribute("data-position", p.x + " " + p.y + " " + p.z);
    hs.setAttribute("data-normal", hit.normal ? (hit.normal.x + " " + hit.normal.y + " " + hit.normal.z) : "0 1 0");
    hs.textContent = "";
    mv.appendChild(hs);
    last = { point: n, pid: "__PID__", layers: meta.layers,
             mesh_grid_index: grid.map(function (v) { return Math.round(v * 10) / 10; }),
             grid_percent_of_mesh: pct,
             approx_full_volume_index: full,
             model_position_m: [p.x, p.y, p.z].map(function (v) { return Math.round(v * 10000) / 10000; }),
             note: meta.coordinate_note };
    box.textContent = "Point " + n +
      "\\n  mesh grid (a0, a1, a2) : " + last.mesh_grid_index.join(", ") +
      (pct ? "\\n  % across mesh grid    : " + pct.join(", ") : "") +
      (full ? "\\n  approx full-vol index : " + full.join(", ") + "  (linear rescale — orientation only)" : "") +
      "\\n  model position (m)    : " + last.model_position_m.join(", ") +
      "\\n  layers in view        : " + meta.layers.join(", ");
  });
  document.getElementById("copyCoords").addEventListener("click", function () {
    if (last && navigator.clipboard) navigator.clipboard.writeText(JSON.stringify(last, null, 1));
  });
  document.getElementById("clearCoords").addEventListener("click", function () {
    mv.querySelectorAll(".hotspot").forEach(function (h) { h.remove(); });
    n = 0; last = null; box.textContent = "Cleared. Tap the model for a new point.";
  });
})();
</script>
</body></html>"""

_ARXR_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PID__ -- WebXR AR with coordinates</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#0d1117;color:#e6edf3}
 a{color:#58a6ff}.hint{color:#8b949e;font-size:14px;line-height:1.5}
 button{background:#e5534b;color:#fff;border:0;border-radius:8px;padding:12px 18px;font-size:16px;cursor:pointer}
 button.secondary{background:#30363d}button:disabled{opacity:.45;cursor:not-allowed}
 #overlay{padding:12px;color:#fff;font-family:system-ui,sans-serif}
 #panel{background:rgba(10,14,20,.82);border:1px solid rgba(255,255,255,.22);border-radius:12px;
        padding:12px;max-width:92vw;font-size:14px;line-height:1.45}
 #readout{font:12.5px ui-monospace,monospace;white-space:pre-wrap;margin:10px 0;max-height:38vh;overflow:auto}
 #overlay button{padding:9px 12px;font-size:14px;margin:3px 4px 0 0}
</style>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
</head><body>
<p><a href="/AR/__PID__?__GLBQ__">&larr; regular AR / 3D page</a></p>
<h2>__PID__ &middot; AR with in-AR coordinates</h2>
<p class="hint"><b>Android Chrome + HTTPS only.</b> This experimental WebXR page stays in the browser during AR,
so it can receive taps that the regular native AR view (Scene Viewer / Quick Look) cannot.
Tap a bare table/floor area to place or move the model; tap directly on the model to pin a point and read its coordinates.
Layers: <b>__SET__</b>.</p>
<button id="start" disabled>Checking WebXR AR support…</button>
<p id="status" class="hint">Loading coordinate metadata…</p>
<div id="overlay" style="display:none">
 <div id="panel">
  <b>WebXR AR — __PID__</b><br>
  <span>Tap floor/table to place. Tap model for coordinates.</span>
  <div id="readout">No point selected yet.</div>
  <button id="copy" class="secondary" type="button">Copy JSON</button>
  <button id="clear" class="secondary" type="button">Clear</button>
  <button id="end" type="button">End AR</button>
 </div>
</div>
<script>
/* Watchdog: if the module below dies silently (CDN blocked, import failure),
   say so instead of sitting on "Checking..." forever. */
window.__ARXRStage = "loading Three.js from CDN";
addEventListener("error", function (e) {
  var s = document.getElementById("status");
  if (s) s.textContent = "Startup error at stage [" + window.__ARXRStage + "]: " +
    (e.message || "script failed to load") + " — check the phone's internet connection and open this page in Chrome (not an in-app browser).";
});
addEventListener("unhandledrejection", function (e) {
  var s = document.getElementById("status");
  if (s) s.textContent = "Startup failed at stage [" + window.__ARXRStage + "]: " + (e.reason && e.reason.message || e.reason);
});
setTimeout(function () {
  if (!window.__ARXRReady) {
    var s = document.getElementById("status");
    if (s && !window.__ARXRDone) s.textContent = "Still starting after 10s — stuck at stage [" + window.__ARXRStage +
      "]. Usual causes: the Three.js CDN (unpkg.com) is slow/blocked on this network, or the .glb is still downloading. Wait a few more seconds; if it never moves, try mobile data instead of institute WiFi.";
  }
}, 10000);
</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const PID = "__PID__", QUERY = "__GLBQ__";
const startBtn = document.getElementById('start');
const statusEl = document.getElementById('status');
const overlay = document.getElementById('overlay');
const readout = document.getElementById('readout');
let meta = null, session = null, last = null, pointNo = 0;
let scene, camera, renderer, model = null, reticle = null;
let hitTestSource = null, refSpace = null;
const markers = [];

const fmt = (a, d=1) => a.map(v => Math.round(v * (10 ** d)) / (10 ** d));

function layerOf(obj) {
  let o = obj;
  while (o) {
    if (meta && meta.layers && meta.layers.includes(o.name)) return o.name;
    o = o.parent;
  }
  return "unknown";
}

function show(hitObj, local) {
  pointNo += 1;
  const t = meta.transform, s = t.scale_m_per_grid_unit;
  const grid = [local.x / s + t.grid_centre[0], -local.z / s + t.grid_centre[1], local.y / s + t.grid_lo[2]];
  const pct = meta.mesh_n ? grid.map((v, i) => Math.round(1000 * v / meta.mesh_n[i]) / 10) : null;
  const full = (meta.mesh_n && meta.mesh_full) ? grid.map((v, i) =>
    Math.round(10 * v * (meta.mesh_full[i] - 1) / (meta.mesh_n[i] - 1)) / 10) : null;
  last = {
    point: pointNo, pid: PID, layer: layerOf(hitObj), layers: meta.layers,
    mesh_grid_index: fmt(grid, 1), grid_percent_of_mesh: pct,
    approx_full_volume_index: full,
    model_position_m: fmt([local.x, local.y, local.z], 4),
    note: meta.coordinate_note
  };
  readout.textContent =
`Point ${pointNo}
layer hit             : ${last.layer}
mesh grid (a0,a1,a2)   : ${last.mesh_grid_index.join(', ')}
% across mesh grid     : ${pct ? pct.join(', ') : '—'}
approx full-vol index : ${full ? full.join(', ') : '—'} (linear rescale only)
model position (m)     : ${last.model_position_m.join(', ')}`;

  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.0045, 20, 14),
    new THREE.MeshBasicMaterial({ color: 0xff3b30 })
  );
  marker.name = 'coordinate-marker';
  marker.position.copy(local);
  model.add(marker);
  markers.push(marker);
}

async function init() {
  /* Staged startup: support is checked FIRST (it can never hang the page),
     and every stage updates the status line + the watchdog's stage marker,
     so a stall always names the stage it died on. */
  const stage = (s, msg) => { window.__ARXRStage = s; if (msg) statusEl.textContent = msg; };

  stage('checking WebXR support', 'Three.js loaded. Checking WebXR AR support…');
  if (!('xr' in navigator)) {
    window.__ARXRDone = true;
    statusEl.textContent = 'WebXR is not available in this browser (this is the #1 cause of a dead page: an in-app browser, Samsung Internet, Firefox, or an http:// address). Copy the URL into Android Chrome, served over https.' +
      '\\nDiagnostics: secure context = ' + window.isSecureContext +
      ', address = ' + location.protocol + '//' + location.host +
      ', browser = ' + navigator.userAgent +
      ' — if this says Chrome on Android and the address is your https ngrok URL, update Chrome and "Google Play Services for AR" in the Play Store, then fully close and reopen Chrome.';
    return;
  }
  const ok = await navigator.xr.isSessionSupported('immersive-AR').catch(() => false);
  if (!ok) {
    window.__ARXRDone = true;
    statusEl.textContent = 'Immersive AR is not supported here. Use Chrome on an ARCore Android phone over the https (ngrok) address.';
    return;
  }

  stage('loading coordinate metadata', 'AR is supported. Loading coordinate metadata…');
  try {
    const ac = new AbortController(); setTimeout(() => ac.abort(), 20000);
    meta = await (await fetch(`/api/AR/${encodeURIComponent(PID)}.json?${QUERY}`, { signal: ac.signal })).json();
  } catch (e) {
    window.__ARXRDone = true;
    statusEl.textContent = 'Coordinate metadata failed to load (' + e + ') — the GLB would still work in the regular AR page; check the server is reachable at this address.';
    return;
  }

  stage('loading 3D model', 'Metadata ready. Downloading the 3D model…');
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.01, 100);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(innerWidth, innerHeight);
  renderer.AR.enabled = true;
  document.body.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(1, 2, 1);
  scene.add(key);

  reticle = new THREE.Mesh(
    new THREE.RingGeometry(0.055, 0.07, 32).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 })
  );
  reticle.matrixAutoUpdate = false;
  reticle.visible = false;
  scene.add(reticle);

  try {
    const gltf = await new GLTFLoader().loadAsync(`/api/AR/${encodeURIComponent(PID)}.glb?${QUERY}`);
    model = gltf.scene;
    model.visible = false;
    scene.add(model);
  } catch (e) {
    window.__ARXRDone = true;
    statusEl.textContent = 'Model failed to load: ' + e;
    return;
  }

  const controller = renderer.AR.getController(0);
  controller.addEventListener('select', onSelect);
  scene.add(controller);

  window.__ARXRReady = true; window.__ARXRDone = true; window.__ARXRStage = 'ready';
  startBtn.disabled = false;
  startBtn.textContent = 'Start WebXR AR';
  statusEl.textContent = 'Ready. The model will appear after you tap a detected floor/table area.';
}

function onSelect() {
  if (!model) return;
  const raycaster = new THREE.Raycaster();
  const m = new THREE.Matrix4();
  const controller = renderer.AR.getController(0);
  m.identity().extractRotation(controller.matrixWorld);
  raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
  raycaster.ray.direction.set(0, 0, -1).applyMatrix4(m);

  if (model.visible) {
    const hits = raycaster.intersectObject(model, true)
      .filter(h => h.object.name !== 'coordinate-marker');
    if (hits.length) {
      const local = model.worldToLocal(hits[0].point.clone());
      show(hits[0].object, local);
      return;
    }
  }
  if (reticle && reticle.visible) {
    model.position.setFromMatrixPosition(reticle.matrix);
    model.quaternion.setFromRotationMatrix(reticle.matrix);
    model.visible = true;
    readout.textContent = 'Model placed. Now tap directly on the model for coordinates.';
  }
}

startBtn.addEventListener('click', async () => {
  try {
    overlay.style.display = 'block';
    session = await navigator.xr.requestSession('immersive-AR', {
      requiredFeatures: ['hit-test'],
      optionalFeatures: ['dom-overlay', 'local-floor'],
      domOverlay: { root: overlay }
    });
    renderer.AR.setReferenceSpaceType('local');
    await renderer.AR.setSession(session);
    refSpace = renderer.AR.getReferenceSpace();
    const viewerSpace = await session.requestReferenceSpace('viewer');
    hitTestSource = await session.requestHitTestSource({ space: viewerSpace });
    session.addEventListener('end', () => {
      hitTestSource = null;
      overlay.style.display = 'none';
      statusEl.textContent = 'AR session ended.';
    });
    renderer.setAnimationLoop((time, frame) => {
      if (frame && hitTestSource && refSpace) {
        const results = frame.getHitTestResults(hitTestSource);
        if (results.length) {
          const pose = results[0].getPose(refSpace);
          reticle.visible = true;
          reticle.matrix.fromArray(pose.transform.matrix);
        } else {
          reticle.visible = false;
        }
      }
      renderer.render(scene, camera);
    });
  } catch (e) {
    overlay.style.display = 'none';
    statusEl.textContent = 'Could not start AR: ' + e + ' (check HTTPS, Chrome, and camera permission.)';
  }
});

document.getElementById('end').addEventListener('click', () => { if (session) session.end(); });
document.getElementById('clear').addEventListener('click', () => {
  markers.forEach(x => x.removeFromParent());
  markers.length = 0; pointNo = 0; last = null;
  readout.textContent = 'Cleared. Tap the model for a new point.';
});
document.getElementById('copy').addEventListener('click', () => {
  if (last && navigator.clipboard) navigator.clipboard.writeText(JSON.stringify(last, null, 1));
});

addEventListener('resize', () => {
  if (!renderer) return;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

init();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):                     # keep the console readable
        if "api" not in str(a[0] if a else ""):
            print("  %s" % (fmt % a))

    def _send(self, body, ctype, code=200, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _file(self, path):
        if not path or not os.path.isfile(path):
            return self._send("not found: %s" % os.path.basename(path or ""), "text/plain", 404)
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as fh:
            self._send(fh.read(), MIME.get(ext, "application/octet-stream"))

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        p, query = parsed.path, parsed.query
        if p in ("/favicon.ico", "/favicon.png"):
            return self._send(b"", "image/x-icon", 204)
        if p in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if p == "/api/studies":
            return self._send(json.dumps(_studies()), MIME[".json"])
        if p == "/api/health":
            return self._send(json.dumps({"ok": True, "data": DATA, "t": time.time()}), MIME[".json"])
        # ---- AR routes -------------------------------------------------
        m = re.match(r"^/AR/([\w\-.]+)$", p)
        if m:
            pid = m.group(1)
            if not os.path.isfile(os.path.join(DATA, pid + ".json")):
                return self._send("no such study: %s" % pid, "text/plain", 404)
            try:
                layers = _ar_layers(query)
            except ValueError as e:
                return self._send("bad AR request: %s" % e, "text/plain", 400)
            if layers:
                glbq, label = "layers=" + layers, layers
            else:
                glbq, label = "set=" + _ar_set(query), _ar_set(query)
            page = (_AR_PAGE.replace("__PID__", pid)
                            .replace("__GLBQ__", glbq).replace("__SET__", label))
            return self._send(page, MIME[".html"])
        if p == "/ARXR-test":
            # Bare WebXR capability probe on THIS origin -- no libraries, no model.
            # If this says immersive-ar is supported but /ARXR/<pid> does not, the
            # difference is in the ARXR page itself; if it also says unsupported,
            # the difference vs. other sites is per-origin (site permission/settings).
            test = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>WebXR test</title>
<style>body{font-family:monospace;padding:16px;background:#111;color:#eee;font-size:13px;white-space:pre-wrap}</style>
</head><body id="b">probing…
<script>
(async function () {
  var out = [];
  out.push("secure context : " + window.isSecureContext);
  out.push("address        : " + location.protocol + "//" + location.host);
  out.push("navigator.AR   : " + ("AR" in navigator));
  out.push("user agent     : " + navigator.userAgent);
  if ("AR" in navigator) {
    for (const mode of ["immersive-AR", "immersive-vr", "inline"]) {
      try { out.push(mode + " supported: " + await navigator.AR.isSessionSupported(mode)); }
      catch (e) { out.push(mode + " check THREW: " + e.name + ": " + e.message); }
    }
    try {
      const st = await navigator.permissions.query({ name: "camera" }).catch(e => null);
      out.push("camera permission: " + (st ? st.state : "query unavailable"));
    } catch (e) { out.push("camera permission: " + e.message); }
  }
  document.getElementById("b").textContent = out.join("\n");
})();
</script></body></html>"""
            return self._send(test, MIME[".html"])
        m = re.match(r"^/ARXR/([\w\-.]+)$", p)
        if m:
            pid = m.group(1)
            if not os.path.isfile(os.path.join(DATA, pid + ".json")):
                return self._send("no such study: %s" % pid, "text/plain", 404)
            try:
                layers = _ar_layers(query)
            except ValueError as e:
                return self._send("bad AR request: %s" % e, "text/plain", 400)
            if layers:
                glbq, label = "layers=" + layers, layers
            else:
                glbq, label = "set=" + _ar_set(query), _ar_set(query)
            page = (_ARXR_PAGE.replace("__PID__", pid)
                              .replace("__GLBQ__", glbq).replace("__SET__", label))
            return self._send(page, MIME[".html"])
        m = re.match(r"^/api/AR/([\w\-.]+)\.glb$", p)
        if m:
            try:
                layers = _ar_layers(query)
            except ValueError as e:
                return self._send("bad AR request: %s" % e, "text/plain", 400)
            path = _ar_glb_path(m.group(1), _ar_set(query), layers=layers)
            if not path:
                return self._send("no AR model for %s (payload missing or has no "
                                  "mesh3d layers)" % m.group(1), "text/plain", 404)
            return self._file(path)
        m = re.match(r"^/api/AR/([\w\-.]+)\.json$", p)
        if m:   # coordinate metadata for the tap-to-coordinates feature
            if ar_glb is None:
                return self._send("AR converter unavailable", "text/plain", 501)
            src = _contained(DATA, m.group(1) + ".json")
            if not src or not os.path.isfile(src):
                return self._send("no such study: %s" % m.group(1), "text/plain", 404)
            try:
                layers = _ar_layers(query)
                with open(src, encoding="utf-8") as fh:
                    payload = json.load(fh)
                out = ar_glb.ar_metadata(payload, _ar_set(query), layers=layers)
                out["pid"] = m.group(1)
            except ValueError as e:
                return self._send("bad AR request: %s" % e, "text/plain", 400)
            return self._send(json.dumps(out), MIME[".json"])
        m = re.match(r"^/api/AR/([\w\-.]+)\.usdz$", p)
        if m:   # iOS Quick Look file, if the user converted & dropped one in
            return self._file(_contained(DATA, m.group(1) + ".usdz"))
        # ---- original routes -------------------------------------------
        m = re.match(r"^/api/study/([\w\-.]+)$", p)
        if m:
            return self._file(_contained(DATA, m.group(1) + ".json"))
        m = re.match(r"^/report/([\w\-.]+)$", p)
        if m:
            return self._file(_contained(DATA, m.group(1) + ".html"))
        if p.startswith("/static/"):
            name = os.path.normpath(p[len("/static/"):]).replace("\\", "/")
            if name.startswith("..") or os.path.isabs(name):
                return self._send("no", "text/plain", 400)
            return self._file(_contained(STATIC, name))
        return self._send("not found", "text/plain", 404)


def main():
    global DATA
    ap = argparse.ArgumentParser(description="Local reading-aid app server.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--data", default=DATA,
                    help="payload directory (default: app/data, or $APP_DATA)")
    a = ap.parse_args()
    DATA = a.data
    os.makedirs(DATA, exist_ok=True)
    n = len(_studies())
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("reading aid on http://%s:%d  (%d study/studies in %s)" % (a.host, a.port, n, DATA))
    print("AR: open /AR/<pid> on a phone (HTTPS via tunnel, e.g. ngrok http %d)" % a.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
