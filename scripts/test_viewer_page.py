"""Open a standalone viewer page from file:// in Chrome or Edge and check it (core env, standard library only).

The EXACT html file is opened, unmodified, through the browser's DevTools protocol on a local port. Nothing is
served and nothing leaves the machine. Checks: start-up state (axial, plain CT, how many volumes NiiVue holds),
load time, scrolling, window changes, views, repeated overlay switching, grade entry and CSV export, an overlay
that fails, and WebGL context loss. Memory is the working set summed over the browser's processes for this profile
(Windows), and GPU memory from nvidia-smi when the browser runs on the NVIDIA card.

Usage: python scripts/test_viewer_page.py --page runs/M2_location_first_20261122/viewer_valid_832_v3.html
           [--browser chrome|edge] [--gpu] [--rounds 3] [--label x]
Writes runs/V1_viewer_test_<label>_<time>/{result.json,result.md,startup.png}. Old pages (viewer v1 / v2) get the
subset of checks they can support (start-up, memory, overlay toggling).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BROWSERS = {"chrome": [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                       r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
            "edge": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]}


class CDP:
    """Minimal DevTools-protocol client over a WebSocket (text frames only)."""

    def __init__(self, ws_url: str):
        host, rest = ws_url[len("ws://"):].split("/", 1)
        h, p = host.split(":")
        self.s = socket.create_connection((h, int(p)), timeout=600)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        if b" 101 " not in buf.split(b"\r\n")[0]:
            raise RuntimeError("websocket handshake failed")
        self.rest = buf.split(b"\r\n\r\n", 1)[1]
        self.n = 0

    def _read(self, k: int) -> bytes:
        while len(self.rest) < k:
            d = self.s.recv(1 << 20)
            if not d:
                raise RuntimeError("browser closed the connection")
            self.rest += d
        out, self.rest = self.rest[:k], self.rest[k:]
        return out

    def _recv(self) -> dict:
        data = b""
        while True:
            b0, b1 = self._read(2)
            ln = b1 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(ln)
            if b0 & 0x0F == 0x8:
                raise RuntimeError("browser closed the websocket")
            if b0 & 0x0F in (0x0, 0x1, 0x2):
                data += payload
                if b0 & 0x80:
                    return json.loads(data.decode("utf-8", "replace"))

    def call(self, method: str, **params):
        self.n += 1
        raw = json.dumps({"id": self.n, "method": method, "params": params}).encode()
        mask = os.urandom(4)
        hdr = bytes([0x81]) + (bytes([0x80 | len(raw)]) if len(raw) < 126 else
                               bytes([0x80 | 126]) + struct.pack(">H", len(raw)) if len(raw) < 65536 else
                               bytes([0x80 | 127]) + struct.pack(">Q", len(raw)))
        self.s.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(raw)))
        while True:
            m = self._recv()
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m["result"]

    def js(self, expr: str, timeout_s: int = 300):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True, timeout=timeout_s * 1000)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError("page script failed: " + str((d.get("exception") or {}).get("description") or d.get("text")))
        return r["result"].get("value")


def proc_mem_mb(profile: Path) -> dict:
    """Working set and private bytes (MB) summed over this profile's browser processes, and the largest single process."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*" + profile.name + "*' } | ForEach-Object { "
          "$t = 'browser'; if ($_.CommandLine -match '--type=([a-z-]+)') { $t = $Matches[1] }; "
          "[string]$_.WorkingSetSize + ',' + [string]$_.PrivatePageCount + ',' + $t }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.split()
    ws = pv = big = 0
    kinds = {}
    for line in out:
        a = line.split(",")
        if len(a) < 2 or not a[0].isdigit():
            continue
        w = int(a[0]) / 1e6
        ws += w
        pv += int(a[1]) / 1e6
        big = max(big, w)
        k = a[2] if len(a) > 2 and a[2] else "browser"
        kinds[k] = round(kinds.get(k, 0) + w)
    return {"working_set_mb": round(ws), "private_mb": round(pv), "largest_process_mb": round(big), "by_type_mb": kinds}


def gpu_mb() -> int | None:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return int(o.stdout.split()[0])
    except Exception:
        return None


PREP = r"""
(() => { window.__csv = null; window.__alerts = []; window.confirm = () => true; window.alert = m => window.__alerts.push(String(m));
  const co = URL.createObjectURL; URL.createObjectURL = b => { try { b.text().then(t => window.__csv = t); } catch (e) {} return co.call(URL, b); };
  // A test must never leave a file that looks like a reviewer's export: no download is written, and the reviewer name is a test marker.
  const ac = HTMLAnchorElement.prototype.click; HTMLAnchorElement.prototype.click = function () { if (this.hasAttribute('download')) { window.__blockedDownloads = (window.__blockedDownloads || 0) + 1; return; } return ac.call(this); };
  window.__TEST_REVIEWER = 'AUTOMATED TEST - not a reviewer'; return true; })()
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--browser", choices=list(BROWSERS), default="chrome")
    ap.add_argument("--gpu", action="store_true", help="let the browser use the graphics card (default: software rendering)")
    ap.add_argument("--integrated-gpu", action="store_true", help="use the low-power (integrated) graphics adapter: closest to a laptop without a graphics card")
    ap.add_argument("--rounds", type=int, default=3, help="times each overlay is switched on in the switching test")
    ap.add_argument("--label", default="")
    ap.add_argument("--port", type=int, default=9377)
    ap.add_argument("--no-destructive", action="store_true", help="skip the simulated overlay failure and context loss")
    args = ap.parse_args()
    page = (args.page if args.page.is_absolute() else ROOT / args.page).resolve()
    exe = next((p for p in BROWSERS[args.browser] if Path(p).exists()), None)
    if exe is None:
        raise SystemExit(f"{args.browser} is not installed")
    tag = f"{args.label or page.stem}_{args.browser}_{'igpu' if args.integrated_gpu else 'gpu' if args.gpu else 'sw'}"
    out = ROOT / "runs" / f"V1_viewer_test_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    profile = ROOT / "runs" / "test_logs" / f"browser_profile_{tag}_{os.getpid()}"
    url = page.as_uri()
    flags = [exe, "--headless=new", f"--remote-debugging-port={args.port}", "--remote-allow-origins=*", f"--user-data-dir={profile}",
             "--no-first-run", "--no-default-browser-check", "--window-size=1700,1000", "--disable-extensions"]
    flags += (["--enable-gpu", "--ignore-gpu-blocklist", "--force_low_power_gpu"] if args.integrated_gpu else
              ["--enable-gpu", "--ignore-gpu-blocklist", "--force_high_performance_gpu"] if args.gpu else ["--disable-gpu"])
    R = {"page": str(page.relative_to(ROOT)), "file_mb": round(page.stat().st_size / 1e6, 1), "browser": args.browser, "url_scheme": url.split(":")[0],
         "gpu_mode": "integrated graphics" if args.integrated_gpu else "hardware" if args.gpu else "software", "checks": [], "memory": {}, "timings": {}}

    def ok(name, cond, detail=None):
        R["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail is not None else ""), flush=True)

    g0 = gpu_mb()
    t_launch = time.time()
    proc = subprocess.Popen(flags + [url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    c = None
    try:
        tgt = None
        for _ in range(240):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json", timeout=2).read())
                tgt = next((t for t in tabs if t.get("type") == "page" and t.get("url", "").startswith("file:")), None)
                if tgt:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not tgt:
            raise RuntimeError("the browser did not open the page")
        c = CDP(tgt["webSocketDebuggerUrl"])
        c.call("Runtime.enable")
        R["browser_version"] = c.call("Browser.getVersion")["product"]
        v3 = False
        peak = {}
        state = None
        for _ in range(900):                       # wait for the scan (up to 15 min)
            m = proc_mem_mb(profile)
            for k in ("working_set_mb", "private_mb", "largest_process_mb"):
                peak[k] = max(peak.get(k, 0), m[k])
            try:
                state = c.js("(() => { const v = window.__viewer; if (v) return {v3: true, ready: v.ready, err: document.getElementById('vstat').className === 'err' ? document.getElementById('vstat').textContent : null};"
                             " return {v3: false, ready: document.readyState === 'complete' && document.querySelectorAll('#ovbar .ov').length > 0, err: null}; })()", 30)
            except Exception:
                state = None
            if state and (state["ready"] or state["err"]):
                break
            time.sleep(1.0)
        R["timings"]["launch_to_ready_s"] = round(time.time() - t_launch, 1)
        R["memory"]["peak_while_loading"] = peak
        if not state or not state["ready"]:
            ok("scan loaded", False, (state or {}).get("err") or "timed out")
            raise RuntimeError("scan did not load")
        v3 = state["v3"]
        ok("opened from file:// and the scan loaded", c.js("location.protocol") == "file:", f"{R['timings']['launch_to_ready_s']} s from browser launch")
        time.sleep(3)
        nav = c.js("(() => { const n = performance.getEntriesByType('navigation')[0]; return {dom_s: Math.round(n.domContentLoadedEventEnd) / 1000, load_s: Math.round(n.loadEventEnd) / 1000}; })()")
        R["timings"].update(nav)
        heap = "performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null"
        before_gc = {**proc_mem_mb(profile), "js_heap_mb": c.js(heap)}
        c.call("HeapProfiler.collectGarbage")              # what is still held once garbage is collected
        time.sleep(2)
        R["memory"]["after_startup_before_gc"] = before_gc
        R["memory"]["after_startup"] = {**proc_mem_mb(profile), "js_heap_mb": c.js(heap),
                                        "gpu_used_delta_mb": None if g0 is None else gpu_mb() - g0}
        (out / "startup.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
        c.js(PREP)
        if v3:
            s = c.js("(() => { const v = window.__viewer, nv = v.nv, gl = nv.gl; const dbg = gl.getExtension('WEBGL_debug_renderer_info');"
                     " return {volumes: nv.volumes.length, active: v.active, chipsOn: document.querySelectorAll('#ovbar .ov:not(.off)').length, chips: document.querySelectorAll('#ovbar .ov').length,"
                     " axial: nv.opts.sliceType === nv.sliceTypeAxial, render: nv.opts.multiplanarShowRender === niivue.SHOW_RENDER.NEVER, verified: v.verified, timings: v.timings,"
                     " ctBlocksLeft: document.querySelectorAll('script.vp[data-vol=ct]').length, overlayBlocks: document.querySelectorAll('script.vp:not([data-vol=ct])').length,"
                     " errors: window.__viewerErrors, secure: window.isSecureContext, renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),"
                     " dims: nv.volumes[0].hdr.dims.slice(1, 4), dtype: nv.volumes[0].hdr.datatypeCode, calmin: nv.volumes[0].cal_min, calmax: nv.volumes[0].cal_max,"
                     " max3d: gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)}; })()")
            R["startup"] = s
            R["timings"].update({"page_" + k: v for k, v in s["timings"].items()})
            ok("NiiVue holds exactly one volume at start-up (the CT; no overlay loaded)", s["volumes"] == 1, f"volumes={s['volumes']}")
            ok("no overlay switched on, no overlay payload decoded", s["chipsOn"] == 0 and s["active"] is None, f"{s['chips']} overlay buttons, all off; {s['overlayBlocks']} compressed blocks untouched")
            ok("axial view at start-up", s["axial"])
            ok("no 3D rendering pane", s["render"])
            ok("lung window at start-up", s["calmin"] == -1350 and s["calmax"] == 150)
            ok("scan verified against the build-time checksum" + (" and SHA-256" if s["verified"] and s["verified"]["sha256"] is True else ""),
               bool(s["verified"]) and s["verified"]["checksum"] is True, str(s["verified"]))
            ok("compressed CT text removed from the page after decoding", s["ctBlocksLeft"] == 0)
            ok("no viewer error at start-up", not s["errors"], s["errors"] or None)
            R["renderer"] = s["renderer"]
            # scrolling (mouse wheel on the canvas) and window buttons
            before = c.js("Array.from(window.__viewer.nv.scene.crosshairPos)")
            box = c.js("(() => { const r = document.getElementById('gl').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()")
            for _ in range(6):
                c.call("Input.dispatchMouseEvent", type="mouseWheel", x=box[0], y=box[1], deltaX=0, deltaY=-120)
                time.sleep(0.15)
            after = c.js("Array.from(window.__viewer.nv.scene.crosshairPos)")
            ok("mouse wheel moves through the slices", abs(after[2] - before[2]) > 1e-4, f"slice position {before[2]:.3f} -> {after[2]:.3f}")
            w = c.js("(() => { const out = []; for (const b of document.querySelectorAll('[data-w]')) { b.click(); const v = window.__viewer.nv.volumes[0]; out.push([b.textContent, v.cal_min, v.cal_max]); } document.querySelector('[data-w]').click(); return out; })()")
            ok("window buttons change the display range", [x[1:] for x in w] == [[-1350, 150], [-160, 240], [-500, 1300]], str(w))
            v = c.js("(() => { const nv = window.__viewer.nv, out = []; for (const b of document.querySelectorAll('[data-v]')) { b.click(); out.push(nv.opts.sliceType); } document.querySelector('[data-v]').click(); return out.concat([nv.opts.sliceType]); })()")
            ok("coronal, sagittal and 3-plane views available; back to axial", v == [0, 1, 2, 3, 0], str(v))
            # repeated overlay switching
            keys = c.js("Array.from(document.querySelectorAll('#ovbar .ov')).map(e => e.dataset.k)")
            maxvol, bad, n_sw = 1, [], 0
            t0 = time.time()
            for r in range(args.rounds):
                for k in keys:
                    st = c.js(f"(async () => {{ document.querySelector('#ovbar .ov[data-k=\"{k}\"]').click(); await window.__viewer.idle(); const v = window.__viewer;"
                              " return {vol: v.nv.volumes.length, active: v.active, on: document.querySelectorAll('#ovbar .ov:not(.off)').length, err: window.__viewerErrors.length}; })()", 300)
                    n_sw += 1
                    maxvol = max(maxvol, st["vol"])
                    if st["vol"] != 2 or st["active"] != k or st["on"] != 1 or st["err"]:
                        bad.append((k, st))
                if r == 0:
                    R["memory"]["after_first_round"] = {**proc_mem_mb(profile), "gpu_used_delta_mb": None if g0 is None else gpu_mb() - g0}
            R["timings"]["overlay_switch_mean_s"] = round((time.time() - t0) / max(n_sw, 1), 2)
            ok(f"{n_sw} overlay switches: always the CT plus exactly one overlay", not bad and maxvol == 2, f"max volumes held = {maxvol}" + (f"; problems {bad[:2]}" if bad else ""))
            # window change and a second click while an overlay is on (the calls that leak a texture in NiiVue 0.69.0)
            c.js("(async () => { for (let i = 0; i < 6; i++) for (const b of document.querySelectorAll('[data-w]')) { b.click(); await new Promise(r => setTimeout(r, 50)); } document.querySelector('[data-w]').click(); return true; })()", 300)
            R["memory"]["after_switching"] = {**proc_mem_mb(profile), "js_heap_mb": c.js("performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null"),
                                              "gpu_used_delta_mb": None if g0 is None else gpu_mb() - g0, "overlay_textures_reused": c.js("window.__viewer.texReused || 0")}
            jump = c.js("(async () => { const tr = document.querySelector('tr.jump'); if (!tr) return null; tr.click(); await new Promise(r => setTimeout(r, 200)); await window.__viewer.idle(); await new Promise(r => setTimeout(r, 100));"
                        " return {key: tr.dataset.jump, active: window.__viewer.active, sel: tr.classList.contains('sel'), vol: window.__viewer.nv.volumes.length}; })()", 300)
            staged1 = c.js("document.body.classList.contains('stage1')")
            if jump and staged1:   # staged page, step 1 not locked: the tool's outlines must not be reachable from the rows
                ok("staged page, step 1: 'Show in scan' does nothing before the lock", jump["active"] != jump["key"] or not jump["sel"], str(jump))
            elif jump:
                ok("'Show in scan' switches on that outline only", jump["active"] == jump["key"] and jump["sel"] and jump["vol"] == 2, str(jump))
            off = c.js("(async () => { const v = window.__viewer; document.querySelector('#ovbar .ov:not(.off)').click(); await v.idle(); return {vol: v.nv.volumes.length, active: v.active, on: document.querySelectorAll('#ovbar .ov:not(.off)').length}; })()", 120)
            ok("switching the overlay off returns to the plain CT (one volume)", off == {"vol": 1, "active": None, "on": 0}, str(off))
            time.sleep(2)
            R["memory"]["after_all_overlays_off"] = {**proc_mem_mb(profile), "gpu_used_delta_mb": None if g0 is None else gpu_mb() - g0}
        else:
            s = c.js("({chipsOn: document.querySelectorAll('#ovbar .ov:not(.off)').length, chips: document.querySelectorAll('#ovbar .ov').length})")
            ok("no overlay switched on at start-up", s["chipsOn"] == 0, f"{s['chips']} overlay buttons")
            keys = c.js("Array.from(document.querySelectorAll('#ovbar .ov')).map(e => e.dataset.k)")
            n_sw = 0
            for r in range(args.rounds):
                for k in keys:
                    c.js(f"(async () => {{ document.querySelector('#ovbar .ov[data-k=\"{k}\"]').click(); await new Promise(r => setTimeout(r, 2500)); return true; }})()", 120)
                    n_sw += 1
                if r == 0:
                    R["memory"]["after_first_round"] = {**proc_mem_mb(profile), "gpu_used_delta_mb": None if g0 is None else gpu_mb() - g0}
            c.js("(async () => { for (let i = 0; i < 6; i++) for (const b of document.querySelectorAll('[data-w]')) { b.click(); await new Promise(r => setTimeout(r, 50)); } return true; })()", 300)
            R["memory"]["after_switching"] = {**proc_mem_mb(profile), "js_heap_mb": c.js("performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null"),
                                              "gpu_used_delta_mb": None if g0 is None else gpu_mb() - g0}
            ok(f"{n_sw} overlay clicks completed without the page dying", c.js("document.querySelectorAll('#ovbar .ov').length") == len(keys))
        # grading and CSV export (both page generations)
        g = c.js("(async () => { const rev = document.getElementById('grev'); rev.value = window.__TEST_REVIEWER; rev.oninput();"
                 " const rows = Array.from(document.querySelectorAll('#chk tr')).slice(1); let lobes = 0; const prefilled = !!document.querySelector('#chk td.grade .gb button.sel');"
                 " rows.forEach((tr, i) => { const td = tr.querySelector('td.grade'); const want = td.querySelector('.gb button.sel') || td.querySelector('.gb button[data-a=\"' + ['present', 'absent', 'cannot_tell'][i % 3] + '\"]'); if (!want.classList.contains('sel')) want.click();"
                 "   const lb = td.querySelector('.lb[data-l=\"RLL\"]'); if (i === 0 && lb && !lb.closest('.gl').classList.contains('idle') && !lb.classList.contains('sel')) { lb.click(); lobes++; } });"
                 " const lock = document.getElementById('glock'); if (lock) lock.click();"
                 " document.getElementById('gsave').click(); await new Promise(r => setTimeout(r, 1200));"
                 " const csv = window.__csv || ''; const lines = csv.replace(/^\\uFEFF/, '').split('\\r\\n');"
                 " return {rows: rows.length, lobes, lines: lines.length, head: lines[0], first: lines[1], graded: document.getElementById('gcount').textContent, alerts: window.__alerts, prefilled}; })()", 120)
        R["grading"] = {k: g[k] for k in ("rows", "lobes", "lines", "graded", "alerts")}
        ok("grades entered for every condition and autosaved", g["graded"].startswith(f"{g['rows']} of {g['rows']}"), g["graded"])
        ok("CSV export: header + one row per condition + overall row", g["lines"] == g["rows"] + 2 and g["head"].startswith("case_id,reviewer,saved_utc,condition_key"), f"{g['lines']} lines")
        ok("CSV carries the grade" + (" and the lobe" if g["lobes"] else ""), '"AUTOMATED TEST - not a reviewer"' in g["first"] and ('"present"' in g["first"] or g["prefilled"]) and (not g["lobes"] or '"RLL"' in g["first"]) or g["prefilled"], g["first"][:160])
        if v3 and not args.no_destructive:
            # an overlay whose data is damaged must be reported and must not block the next click
            f = c.js("(async () => { const v = window.__viewer, keys = Array.from(document.querySelectorAll('#ovbar .ov')).map(e => e.dataset.k);"
                     " const el = document.querySelector('script.vp[data-vol=\"' + keys[0] + '\"]'); const keep = el.textContent; el.textContent = keep.slice(0, 200) + '!!!!' + keep.slice(204);"
                     " const n0 = window.__viewerErrors.length; document.querySelector('#ovbar .ov[data-k=\"' + keys[0] + '\"]').click(); await v.idle();"
                     " const a = {active: v.active, vol: v.nv.volumes.length, warned: window.__viewerErrors.length > n0, banner: document.getElementById('vbanner').textContent, failMark: document.querySelector('#ovbar .ov[data-k=\"' + keys[0] + '\"]').classList.contains('fail')};"
                     " document.querySelector('#ovbar .ov[data-k=\"' + keys[1] + '\"]').click(); await v.idle(); a.next = {active: v.active, vol: v.nv.volumes.length, want: keys[1]};"
                     " el.textContent = keep; document.querySelector('#ovbar .ov[data-k=\"' + keys[0] + '\"]').click(); await v.idle(); a.retry = v.active === keys[0];"
                     " document.querySelector('#ovbar .ov:not(.off)').click(); await v.idle(); return a; })()", 300)
            ok("a damaged overlay is reported on the page", f["warned"] and f["active"] is None and f["vol"] == 1 and f["failMark"], f["banner"][:150])
            ok("after that failure the next overlay still loads, and the failed one can be retried", f["next"]["active"] == f["next"]["want"] and f["next"]["vol"] == 2 and f["retry"], str(f["next"]))
            lost = c.js("(async () => { const v = window.__viewer, ext = v.nv.gl.getExtension('WEBGL_lose_context'); if (!ext) return null; ext.loseContext(); await new Promise(r => setTimeout(r, 800));"
                        " const st = document.getElementById('vstat'); const td = document.querySelector('#chk td.grade'); const b = td.querySelector('.gb button[data-a=\"absent\"]'); b.click();"
                        " return {lost: v.lost, shown: getComputedStyle(st).display !== 'none' && st.className === 'err', text: st.textContent.slice(0, 120), gradeStillWorks: b.classList.contains('sel'),"
                        " reload: !!st.querySelector('button')}; })()", 60)
            if lost is None:
                ok("WebGL context loss handled", False, "WEBGL_lose_context not available in this browser mode")
            else:
                ok("WebGL context loss shows a message with a reload button", lost["lost"] and lost["shown"] and lost["reload"], lost["text"])
                ok("grading still works after the viewer stops", lost["gradeStillWorks"])
    except Exception as e:  # reported, never dropped
        R["error"] = f"{type(e).__name__}: {e}"
        print("ERROR", R["error"], flush=True)
    finally:
        try:
            if c:
                c.call("Browser.close")
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*" + profile.name + "*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                       capture_output=True)
        time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)
    R["passed"] = "error" not in R and all(x["pass"] for x in R["checks"])
    (out / "result.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    md = [f"# Viewer page test: {R['page']} ({R['file_mb']} MB) in {R.get('browser_version', args.browser)} - {'PASSED' if R['passed'] else 'FAILED'}", "",
          f"Opened from file:// on the development workstation, {R['gpu_mode']} rendering" + (f" ({R.get('renderer')})" if R.get("renderer") else "") + ".", "",
          "| check | result | detail |", "|---|---|---|"]
    md += [f"| {x['name']} | {'pass' if x['pass'] else 'FAIL'} | {str(x['detail'] or '')[:200].replace('|', '/')} |" for x in R["checks"]]
    md += ["", "Timings: " + json.dumps(R["timings"]), "", "Memory (MB): " + json.dumps(R["memory"])]
    if "error" in R:
        md += ["", "ERROR: " + R["error"]]
    (out / "result.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"timings": R["timings"], "memory": R["memory"]}, indent=1))
    print(f"-> {out}")
    return 0 if R["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
