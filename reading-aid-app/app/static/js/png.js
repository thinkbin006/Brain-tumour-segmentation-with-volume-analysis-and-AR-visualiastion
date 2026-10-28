/* png.js -- minimal dependency-free PNG decoder (8-bit, non-interlaced).
 * N3 packs the coarse volumes as tall grayscale PNGs and the slice stacks as RGBA PNGs;
 * decoding them ourselves (instead of via <canvas>) gives exact byte values in BOTH the
 * browser and the node test harness, with no colour-management surprises.
 * Inflate uses the platform DecompressionStream('deflate') (browser + node >= 18). */
(function (root) {
  "use strict";
  var RA = root.RA || (root.RA = {});

  function b64ToBytes(s) {
    var i, out;
    if (typeof atob === "function") {
      var bin = atob(s);
      out = new Uint8Array(bin.length);
      for (i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(s, "base64"));           /* node */
  }
  function uriToBytes(uri) {
    var k = uri.indexOf(",");
    return b64ToBytes(k >= 0 ? uri.slice(k + 1) : uri);
  }
  async function inflate(bytes) {
    /* no Blob dependency: Blob.stream() is missing in some non-browser runtimes, and this
       runs identically in the browser and in the node test harness */
    var ds = new DecompressionStream("deflate");                /* zlib wrapper */
    var rs = new ReadableStream({ start: function (c) { c.enqueue(bytes); c.close(); } });
    var resp = new Response(rs.pipeThrough(ds));
    return new Uint8Array(await resp.arrayBuffer());
  }

  var CHAN = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

  /* decode a PNG (Uint8Array file bytes or data URI) -> {width,height,channels,data}
     channels: 1 gray, 2 gray+a, 3 rgb, 4 rgba (palette expanded to 3/4) */
  async function decode(src) {
    var b = (typeof src === "string") ? uriToBytes(src) : src;
    var sig = [137, 80, 78, 71, 13, 10, 26, 10];
    for (var i = 0; i < 8; i++) if (b[i] !== sig[i]) throw new Error("png: bad signature");
    var p = 8, ihdr = null, idat = [], plte = null, trns = null;
    var dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    while (p + 8 <= b.length) {
      var len = dv.getUint32(p); var type = String.fromCharCode(b[p + 4], b[p + 5], b[p + 6], b[p + 7]);
      var body = b.subarray(p + 8, p + 8 + len);
      if (type === "IHDR") {
        ihdr = { w: dv.getUint32(p + 8), h: dv.getUint32(p + 12), depth: b[p + 16],
                 ctype: b[p + 17], interlace: b[p + 20] };
      } else if (type === "IDAT") idat.push(body);
      else if (type === "PLTE") plte = body;
      else if (type === "tRNS") trns = body;
      else if (type === "IEND") break;
      p += 12 + len;
    }
    if (!ihdr) throw new Error("png: no IHDR");
    if (ihdr.depth !== 8) throw new Error("png: only 8-bit supported (got depth " + ihdr.depth + ")");
    if (ihdr.interlace !== 0) throw new Error("png: interlaced not supported");
    var nch = CHAN[ihdr.ctype];
    if (!nch) throw new Error("png: colour type " + ihdr.ctype + " not supported");

    var raw = await inflate(concat(idat));
    var w = ihdr.w, h = ihdr.h, stride = w * nch;
    var out = new Uint8Array(stride * h);
    var prev = new Uint8Array(stride);
    var q = 0;
    for (var y = 0; y < h; y++) {
      var ft = raw[q++];
      var line = raw.subarray(q, q + stride); q += stride;
      var cur = out.subarray(y * stride, (y + 1) * stride);
      unfilter(ft, line, prev, cur, nch);
      prev = cur;
    }
    if (ihdr.ctype === 3) {                                   /* palette -> rgb(a) */
      var hasA = trns && trns.length;
      var och = hasA ? 4 : 3;
      var exp = new Uint8Array(w * h * och);
      for (i = 0; i < w * h; i++) {
        var pi = out[i] * 3;
        exp[i * och] = plte[pi]; exp[i * och + 1] = plte[pi + 1]; exp[i * och + 2] = plte[pi + 2];
        if (hasA) exp[i * och + 3] = (i % w < trns.length) ? trns[out[i]] : 255;
        else if (och === 4) exp[i * och + 3] = 255;
      }
      return { width: w, height: h, channels: och, data: exp };
    }
    return { width: w, height: h, channels: nch, data: out };
  }

  function unfilter(ft, line, prev, out, bpp) {
    var i, a, bb, c, x;
    switch (ft) {
      case 0: out.set(line); break;
      case 1:
        for (i = 0; i < line.length; i++) { a = i >= bpp ? out[i - bpp] : 0; out[i] = (line[i] + a) & 255; }
        break;
      case 2:
        for (i = 0; i < line.length; i++) out[i] = (line[i] + prev[i]) & 255;
        break;
      case 3:
        for (i = 0; i < line.length; i++) { a = i >= bpp ? out[i - bpp] : 0; out[i] = (line[i] + ((a + prev[i]) >> 1)) & 255; }
        break;
      case 4:
        for (i = 0; i < line.length; i++) {
          a = i >= bpp ? out[i - bpp] : 0; bb = prev[i]; c = i >= bpp ? prev[i - bpp] : 0;
          x = a + bb - c;
          var pa = Math.abs(x - a), pb = Math.abs(x - bb), pc = Math.abs(x - c);
          out[i] = (line[i] + (pa <= pb && pa <= pc ? a : (pb <= pc ? bb : c))) & 255;
        }
        break;
      default: throw new Error("png: bad filter " + ft);
    }
  }
  function concat(parts) {
    var n = 0, i;
    for (i = 0; i < parts.length; i++) n += parts.length;
    n = 0; for (i = 0; i < parts.length; i++) n += parts[i].length;
    var out = new Uint8Array(n), o = 0;
    for (i = 0; i < parts.length; i++) { out.set(parts[i], o); o += parts[i].length; }
    return out;
  }

  /* convenience: alpha plane of an RGBA png as Uint8Array(w*h) (or gray values) */
  async function alphaOf(uri) {
    var im = await decode(uri);
    if (im.channels === 1) return im.data;
    var out = new Uint8Array(im.width * im.height);
    var step = im.channels;
    for (var i = 0; i < out.length; i++) out[i] = im.data[i * step + (step === 2 ? 0 : step - 1)];
    return out;
  }
  function pngDims(uri) {
    var b = uriToBytes(uri);
    var dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    return { width: dv.getUint32(16), height: dv.getUint32(20), depth: b[24], ctype: b[25] };
  }

  RA.PNG = { decode: decode, alphaOf: alphaOf, dims: pngDims, b64ToBytes: b64ToBytes, uriToBytes: uriToBytes };
  if (typeof module !== "undefined" && module.exports) module.exports = RA;
})(typeof window !== "undefined" ? window : globalThis);
