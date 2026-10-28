"""Local reading-aid app -- server (v2).

Serves the workstation UI and the study payloads that the notebook (N3) produced.  Nothing here is
a model or a metric: it is the front end.  The producer stays the notebook; this app renders every
section of `report_<pid>.json` and says out loud (in the Payload map drawer) if a section is
missing or unused.

    python3 app/server.py [--port 8000] [--host 0.0.0.0] [--data app/data]

Endpoints
    /                       the workstation
    /static/*               js, css, vendored plotly
    /api/studies            one line per study (id, headline numbers, difficulty tag)
    /api/study/<pid>        the full payload the notebook wrote for that study
    /report/<pid>           the self-contained HTML report for that study (download/inspect)

Changes vs the previous server.py
    * `--data` now exists (the docstring always promised it; argparse never defined it).
    * `/api/studies` caches each payload summary keyed on (mtime, size) -- it used to fully parse
      every study JSON on every worklist refresh.
    * pid routes are realpath-contained inside the data dir (no `..` games, on any OS).
    * favicon answers 204 instead of a 404 text body; .ico/.map/.wasm MIME added.
    * `patient` falls back to the pid server-side, so the worklist never shows "null".
"""
import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "app", "static")
DATA = os.environ.get("APP_DATA") or os.path.join(ROOT, "app", "data")

MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".map": "application/json; charset=utf-8",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml", ".ico": "image/x-icon", ".webp": "image/webp",
        ".nii": "application/octet-stream", ".gz": "application/gzip",
        ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf"}

# what each study is here to show -- the app labels the case list with it, so the demo has a story
NOTES = {
    "satellite":   ("missed satellite", "a second small lesion across the midline that the model does not find"),
    "lowcontrast": ("low contrast", "a faint, patchy lesion: the threshold sits inside the noise"),
    "multifocal":  ("multifocal", "two lesions the model finds but BRIDGES into one"),
    "typical":     ("single lesion", "a plain enhancing lesion with an oedematous rim"),
}

# ---------------------------------------------------------------- summary cache
# _studies() used to json.load EVERY payload on EVERY request; with multi-megabyte payloads that is
# seconds of work per worklist refresh.  Summaries are tiny and payloads are immutable once
# installed, so cache per file, invalidated by (mtime, size).
_CACHE = {}
_CACHE_LOCK = threading.Lock()


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
        # The manifest owns the user-facing label.  A real case has no `variant` (that vocabulary is
        # for the constructed phantoms), so an explicit `tag` wins, then the variant's note, then a
        # neutral default -- nothing user-visible is ever derived from a payload field N3 does not
        # write (that is how every study once read "single lesion").
        tag = mv.get("tag") or note[0]
        why = mv.get("why") or note[1]
        out.append(dict(pid=pid, tag=tag, why=why,
                        patient=mv.get("patient") or pid,
                        variant=mv.get("variant"),
                        has_report=os.path.exists(os.path.join(DATA, pid + ".html")), **s))
    return out


def _contained(directory, name):
    """Resolve `name` inside `directory`; return None if it escapes (any OS, any encoding trick)."""
    p = os.path.realpath(os.path.join(directory, name))
    root = os.path.realpath(directory)
    if p == root or p.startswith(root + os.sep):
        return p
    return None


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
        p = self.path.split("?")[0]
        if p in ("/favicon.ico", "/favicon.png"):
            return self._send(b"", "image/x-icon", 204)
        if p in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if p == "/api/studies":
            return self._send(json.dumps(_studies()), MIME[".json"])
        if p == "/api/health":
            return self._send(json.dumps({"ok": True, "data": DATA, "t": time.time()}), MIME[".json"])
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
    srv.serve_forever()


if __name__ == "__main__":
    main()
