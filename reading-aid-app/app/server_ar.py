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
No new dependencies: ar_glb.py is stdlib-only, and the GLB is generated from the mesh3d
section N3 already writes -- no re-segmentation, no marching cubes at serve time.

Run it exactly as before:
    python3 app/server.py [--port 8000] [--host 0.0.0.0] [--data app/data]

For AR on a real phone the page must be HTTPS (the Google AR app will only fetch the
model over HTTPS; localhost is exempt but a phone is not localhost). Tunnel your usual port, e.g.:   ngrok http 8000
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
        sig = "%d:%d:%s:v%s" % (st.st_mtime_ns, st.st_size, layers or layer_set, getattr(ar_glb, "FORMAT_VERSION", 1))
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
  #startAR{background:#e5534b;color:#fff;border:0;border-radius:8px;padding:14px 20px;
             font-size:17px;cursor:pointer}
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
    &middot; <a href="/api/AR/__PID__.glb?__GLBQ__">download .glb</a></p>

  <model-viewer id="mv" src="/api/AR/__PID__.glb?__GLBQ__"
                alt="3D brain tumour segmentation for __PID__"
                ar ar-modes="webxr" ar-scale="auto" ar-placement="floor"
                camera-controls auto-rotate shadow-intensity="1" exposure="1.1">
    <button slot="ar-button" id="ar-button" style="display:none" aria-hidden="true" tabindex="-1"></button>
  </model-viewer>
  <p><button id="startAR" type="button">&#128247; View in your space (AR)</button>
     <span id="arState" class="hint">checking AR support…</span></p>
  <p class="hint">AR runs inside Chrome (WebXR) — point at the floor/table, tap to place,
     pinch to resize. To keep a memory: screenshot (Power + Volume-down) or screen-record
     while in AR; the 📸 button below saves a photo of the 3D view.</p>

  <p><label class="pick"><input type="checkbox" id="pickMode"> 📍 Tap-to-coordinates
     <span class="hint">— then tap (don't drag) any point on the model in the 3D view above.
     Works in the 3D view above; in AR the camera view is live, so point-tapping
     coordinates is only offered here.</span></label></p>
  <div id="coords">Coordinates will appear here. Turn on tap-to-coordinates and tap the tumour.</div>
  <p><button id="copyCoords" type="button">Copy coordinates (JSON)</button>
     <button id="clearCoords" type="button">Clear points</button>
     <button id="snapPhoto" type="button">📸 Save photo of 3D view</button></p>
  <p class="hint">Crop-local coordinates, as in the .nii export -- no world orientation is
     claimed. Layers: <b>__SET__</b>. Frozen N3 meshes at the certified &lambda;* (a live-&lambda;
     re-threshold exists only as coarse volumes in the payload, so it cannot be re-meshed
     faithfully here).</p>
<script>
(function () {
  var mv = document.getElementById("mv"), box = document.getElementById("coords");
  /* Photo of the 3D view (PNG download). In AR, use the phone screenshot /
     screen-record — Chrome cannot hand the live camera feed to the page. */
  /* Always-visible AR launcher (the viewer's own button only appears after
     its capability check passes, which made it look missing). */
  const arState = document.getElementById("arState");
  (async () => {
    if (!window.isSecureContext || !navigator.xr) {
      arState.textContent = "— WebXR not available in this browser (use Chrome on the phone, via the HTTPS address)";
      return;
    }
    try {
      const ok = await navigator.xr.isSessionSupported("immersive-ar");
      arState.textContent = ok ? "— AR ready: tap the button, point at the floor, tap to place"
                               : "— immersive AR not supported on this device/browser";
    } catch (e) { arState.textContent = "— AR check failed: " + e.message; }
  })();
  document.getElementById("startAR").addEventListener("click", () => {
    if (mv.activateAR) mv.activateAR();
  });
  document.getElementById("snapPhoto").addEventListener("click", async function () {
    try {
      const blob = await mv.toBlob({mimeType: "image/png", idealAspect: true});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = "__PID___3d.png"; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    } catch (e) { box.textContent = "photo failed: " + e; }
  });
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
