#!/usr/bin/env python
"""Minimal localhost web viewer for the SAVED A0 / A-ft / B / C counterfactual results (research demonstration).

    python app/web_demo.py --phase6-op "genai phase 6  v2 op" --phase7-op "genai phase 7 op"      # then open http://127.0.0.1:8765

It only DISPLAYS outputs that were already generated and saved by the Kaggle runs (original, reconstruction, counterfactual) with the stored
scores. It runs no model, regenerates nothing and needs no GPU, no network and no extra dependency (Python standard library + PIL, already
required). Everything it offers is derived from the artifacts it finds: a case, condition or arm that is not in them is shown as unavailable.
It binds to 127.0.0.1 only (the images are CheXpert-derived). The command-line demo (app/demo.py) is unchanged.
"""
import argparse
import io
import json
import math
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import demo as cli  # noqa: E402  (existing CLI demo: disclaimers, arm titles/findings, reviewer notes)

NOTICE = ("Research demonstration using synthetic counterfactual medical images. Not clinically validated. "
          "Not for diagnosis or medical decision-making. Classifier scores do not establish clinical correctness.")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
KINDS = {"original": "original.png", "reconstruction": "reconstructed.png", "counterfactual": "counterfactual.png"}
OTHER_FINDINGS = ("Pneumonia", "Pleural Effusion", "No Finding")        # the project's target findings that the saved results do not cover
METRICS = (  # key, CSV column, label, unit/format
    ("p_orig", "p_target_orig", "p({p}) of the original input", "prob"), ("p_recon", "p_target_recon", "p({p}) of the reconstruction", "prob"),
    ("p_edit", "p_target_cf", "p({p}) after the edit", "prob"), ("dp", "dp_target", "change (edit − reconstruction)", "signed"),
    ("psnr_edit", "psnr_recon_cf", "PSNR(reconstruction, edit)", "db"), ("ssim_edit", "ssim_recon_cf", "SSIM(reconstruction, edit)", "ssim"),
    ("ssim_edit_orig", "ssim_orig_cf", "SSIM(original, edit)", "ssim"), ("psnr_inv", "psnr_orig_recon", "inversion fidelity PSNR(original, reconstruction)", "db"),
    ("changed", "change_frac_changed_gt20", "pixels changed by > 20/255", "pct"), ("brightness", "change_brightness_shift", "mean brightness shift [gray levels]", "signed"),
    ("clip", "clip_directional", "CLIP directional similarity", "ssim"))


class WebDemoError(Exception):
    pass


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _finding(arm: str) -> str:
    """The study report's finding for the arm, without the CLI panel prefix."""
    return cli.ARM_FINDING[arm].replace("report: ", "")


def _condition_key(pathology, operation) -> str:
    return f"{pathology}|{operation}"


class Catalog:
    """Everything the viewer can show, discovered from the saved artifacts (nothing is assumed to exist)."""

    def __init__(self, phase6_op, phase7_op):
        from src.evaluation import phase8 as P8

        roots = {"phase6": Path(phase6_op) if phase6_op else None, "phase7": Path(phase7_op) if phase7_op else None}
        self.arms, self.problems, eval_sets = {}, [], []
        for key, root in roots.items():
            if root is None or not root.is_dir():
                self.problems.append(f"{key} output directory not found ({root if root else 'not given'})")
                continue
            es = root / "phase6" / "eval_set.json"
            if es.exists():
                try:
                    eval_sets.append(json.loads(es.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    self.problems.append(f"unreadable evaluation set: {es}")
        self.eval = eval_sets[0] if eval_sets else {}
        self.meta = {int(e["index"]): e for e in self.eval.get("images", [])}
        for arm, (key, run_name, rel) in P8.ARM_SOURCES.items():
            root = roots[key]
            if root is None or not root.is_dir():
                self.arms[arm] = {"available": False, "reason": f"{key} artifacts not found"}
                continue
            try:
                run_dir = P8.first_run_dir(root, run_name)
            except (FileNotFoundError, OSError):
                self.arms[arm] = {"available": False, "reason": f"run directory {run_name} not found"}
                continue
            scores = None
            if (root / rel).exists():
                try:
                    scores = pd.read_csv(root / rel).set_index("index")
                except (OSError, ValueError, KeyError):
                    self.problems.append(f"unreadable scores for arm {arm}: {root / rel}")
            folders = {}
            out = run_dir / "outputs"
            for d in sorted(out.iterdir()) if out.is_dir() else []:
                try:
                    folders[int(d.name.split("_")[0])] = d
                except ValueError:
                    continue
            self.arms[arm] = {"available": True, "run_dir": run_dir, "scores": scores, "folders": folders, "cond": {}}
        if not any(a["available"] for a in self.arms.values()):
            raise WebDemoError("no saved results found. Pass --phase6-op / --phase7-op (or set PRISM_PHASE6_OP / PRISM_PHASE7_OP); "
                               "problems: " + "; ".join(self.problems or ["none of the expected run directories exist"]))
        # conditions represented by each saved output (from its metadata, else from the evaluation set)
        fallback = (self.eval.get("pathology"), self.eval.get("operation"))
        for arm, a in self.arms.items():
            if not a["available"]:
                continue
            for case, d in a["folders"].items():
                pair = None
                try:
                    m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
                    pair = (m.get("target_pathology"), m.get("operation"))
                except (OSError, ValueError):
                    pass
                if not pair or None in pair:
                    pair = fallback if None not in fallback else None
                a["cond"][case] = _condition_key(*pair) if pair else None
        self.cases = sorted({c for a in self.arms.values() if a["available"] for c in a["folders"]} | set(self.meta))
        self.conditions = sorted({k for a in self.arms.values() if a["available"] for k in a["cond"].values() if k})

    # ------------------------------------------------------------------ availability
    def arm_status(self, arm: str, case: int, condition: str) -> dict:
        a = self.arms.get(arm)
        if a is None:
            return {"available": False, "reason": "unknown arm"}
        if not a["available"]:
            return {"available": False, "reason": a["reason"]}
        d = a["folders"].get(case)
        if d is None:
            return {"available": False, "reason": "no saved output for this case"}
        if a["cond"].get(case) != condition:
            return {"available": False, "reason": "this condition is not represented in the saved output for this case"}
        if not (d / KINDS["counterfactual"]).exists() or not (d / KINDS["original"]).exists():
            return {"available": False, "reason": "saved output incomplete (missing original or counterfactual image)"}
        return {"available": True}

    def image_file(self, arm: str, case: int, kind: str):
        a = self.arms.get(arm)
        if not a or not a["available"] or case not in a["folders"] or kind not in KINDS:
            return None
        p = a["folders"][case] / KINDS[kind]
        return p if p.exists() else None

    # ------------------------------------------------------------------ payloads
    def index_payload(self) -> dict:
        conds = []
        for k in self.conditions:
            pathology, operation = k.split("|")
            conds.append({"key": k, "label": f"{pathology} ({operation})", "available": True})
        shown = {c["key"].split("|")[0] for c in conds}
        for f in OTHER_FINDINGS:
            if f not in shown:
                conds.append({"key": f"{f}|add", "label": f"{f} (add)", "available": False, "reason": "not represented in the saved results (never evaluated in this project)"})
        arms = []
        for arm in cli.ARMS:
            a = self.arms.get(arm, {"available": False, "reason": "unknown"})
            arms.append({"key": arm, "title": cli.ARM_TITLE[arm], "available": a["available"], "reason": a.get("reason"), "finding": _finding(arm),
                         "cases": sum(1 for c in self.cases if self.arm_status(arm, c, self.conditions[0] if self.conditions else "")["available"])})
        return {"notice": NOTICE, "cases": self.cases, "conditions": conds, "arms": arms, "problems": self.problems,
                "evaluation_set": {k: self.eval.get(k) for k in ("split", "operation", "pathology")}}

    def case_payload(self, case: int, condition: str) -> dict:
        if case not in self.cases:
            raise WebDemoError(f"unknown case {case}")
        if condition not in self.conditions:
            raise WebDemoError(f"condition {condition!r} is not represented in the saved results")
        pathology = condition.split("|")[0]
        meta = self.meta.get(case, {})
        labels = {k: v for k, v in (meta.get("labels") or {}).items()}
        out = {"case": case, "condition": condition, "pathology": pathology, "input": {"ap_pa": meta.get("ap_pa"), "labels": labels,
               "note": "labels are CheXpert's automatic-labeler output for the real input image (None = unmentioned)"}, "arms": {}, "notes": cli.case_notes(case)}
        for arm in cli.ARMS:
            st = self.arm_status(arm, case, condition)
            entry = {"available": st["available"], "reason": st.get("reason"), "title": cli.ARM_TITLE[arm], "finding": _finding(arm), "images": {}, "metrics": {}}
            if st["available"]:
                base = f"/img/{arm}/{case}/"
                entry["images"] = {k: (base + k if self.image_file(arm, case, k) else None) for k in KINDS}
                entry["images"]["change"] = base + "change" if entry["images"]["reconstruction"] else None
                row = None
                s = self.arms[arm]["scores"]
                if s is not None and case in s.index:
                    row = s.loc[case]
                for key, col, label, fmt in METRICS:
                    v = _num(row[col]) if (row is not None and col in row.index) else None
                    if key == "changed" and v is not None:
                        v = 100.0 * v
                    entry["metrics"][key] = {"label": label.replace("{p}", pathology), "value": v, "format": fmt}
            out["arms"][arm] = entry
        return out

    def change_png(self, arm: str, case: int):
        e, r = self.image_file(arm, case, "counterfactual"), self.image_file(arm, case, "reconstruction")
        if e is None or r is None:
            return None
        a, b = (np.asarray(Image.open(p).convert("L")).astype(np.int16) for p in (e, r))
        if a.shape != b.shape:
            return None
        buf = io.BytesIO()
        Image.fromarray(np.clip(np.abs(a - b) * 4, 0, 255).astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    catalog: Catalog = None
    server_version = "PrismResearchDemo"

    def log_message(self, fmt, *args):   # keep the console quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        q = parse_qs(url.query)
        try:
            if not parts:
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if parts == ["api", "index"]:
                return self._json(200, self.catalog.index_payload())
            if parts == ["api", "case"]:
                try:
                    case = int(q.get("case", [""])[0])
                except ValueError:
                    return self._json(400, {"error": "case must be an integer"})
                condition = q.get("condition", [""])[0] or (self.catalog.conditions[0] if self.catalog.conditions else "")
                try:
                    return self._json(200, self.catalog.case_payload(case, condition))
                except WebDemoError as e:
                    return self._json(404, {"error": str(e)})
            if len(parts) == 4 and parts[0] == "img":
                arm, kind = parts[1], parts[3]
                try:
                    case = int(parts[2])
                except ValueError:
                    return self._json(400, {"error": "case must be an integer"})
                if kind == "change":
                    data = self.catalog.change_png(arm, case) if arm in cli.ARMS else None
                else:
                    f = self.catalog.image_file(arm, case, kind) if arm in cli.ARMS else None
                    data = f.read_bytes() if f else None
                if data is None:
                    return self._json(404, {"error": "image not available"})
                return self._send(200, data, "image/png")
            return self._json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001 - never crash the server on a bad artifact
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})


def make_server(catalog: Catalog, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise WebDemoError(f"refusing to bind to {host!r}: the viewer serves CheXpert-derived images and listens on the loopback interface only")
    handler = type("BoundHandler", (Handler,), {"catalog": catalog})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase6-op", default=os.environ.get("PRISM_PHASE6_OP"), help="Phase 6 v2 output directory (env PRISM_PHASE6_OP)")
    ap.add_argument("--phase7-op", default=os.environ.get("PRISM_PHASE7_OP"), help="Phase 7 output directory (env PRISM_PHASE7_OP)")
    ap.add_argument("--host", default="127.0.0.1", help="loopback only")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args(argv)
    try:
        catalog = Catalog(args.phase6_op, args.phase7_op)
        server = make_server(catalog, args.host, args.port)
    except WebDemoError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"ERROR: cannot listen on {args.host}:{args.port} ({e}); try --port 0 for a free port", file=sys.stderr)
        return 2
    url = f"http://{args.host}:{server.server_address[1]}"
    print(NOTICE)
    print(f"serving {len(catalog.cases)} saved case(s), arms available: {[a for a, v in catalog.arms.items() if v['available']]}")
    for p in catalog.problems:
        print("note:", p)
    print(f"open {url}   (Ctrl+C to stop)")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Counterfactual X-ray research demo</title>
<style>
:root{--ink:#1d2430;--mut:#5b6573;--line:#d5d9df;--bg:#f4f5f7;--card:#fff;--warn:#8a1c1c;--gen:#2b4a7a}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
#notice{position:sticky;top:0;z-index:10;background:var(--warn);color:#fff;padding:9px 16px;font-weight:600;text-align:center}
main{max-width:1180px;margin:0 auto;padding:18px 16px 40px}h1{font-size:20px;margin:6px 0 2px}.sub{color:var(--mut);margin:0 0 14px}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
label{display:block;font-size:12px;color:var(--mut);margin-bottom:3px}select,button{font:inherit;padding:6px 8px;border:1px solid #b9bfc8;border-radius:6px;background:#fff}
button{cursor:pointer}button:disabled{opacity:.45;cursor:default}.nav{display:flex;gap:6px;align-items:center}
.panels{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}@media(max-width:900px){.panels{grid-template-columns:repeat(2,1fr)}}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}figure img{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#000;filter:grayscale(1)}
figcaption{padding:7px 10px;font-size:13px}.tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.03em;padding:1px 7px;border-radius:10px;margin-right:6px;color:#fff}
.t-in{background:#3b6b3b}.t-syn{background:var(--gen)}.t-map{background:#6b6b6b}.na{color:#8b93a0;font-style:italic}
.missing{display:flex;align-items:center;justify-content:center;aspect-ratio:1/1;background:#e9ebef;color:#8b93a0;font-style:italic;text-align:center;padding:10px}
.grid2{display:grid;grid-template-columns:1.2fr 1fr;gap:14px;margin-top:14px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}.card h2{font-size:14px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}td{padding:4px 6px;border-bottom:1px solid #eceef1}td.v{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.cav{font-size:12.5px;color:var(--mut);margin-top:8px}.strip{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.mini{border:1px solid var(--line);border-radius:6px;padding:6px;cursor:pointer;background:#fff;text-align:center;font-size:12px}.mini.sel{outline:2px solid var(--gen)}.mini.off{opacity:.5;cursor:default}
.mini img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#000;filter:grayscale(1)}footer{color:var(--mut);font-size:12.5px;margin-top:18px}
#err{color:var(--warn);margin-top:10px}
</style></head><body>
<div id="notice" role="alert">Research demonstration using synthetic counterfactual medical images. Not clinically validated. Not for diagnosis or medical decision-making. Classifier scores do not establish clinical correctness.</div>
<main>
<h1>Counterfactual chest X-ray research demo</h1>
<p class="sub">Browse saved results of the A0 / A-ft / B / C experiments. This page only displays outputs that were already generated; it runs no model and cannot regenerate anything.</p>
<div class="controls">
 <div><label for="case">Case</label><div class="nav"><button id="prev" title="previous case">◀</button><select id="case"></select><button id="next" title="next case">▶</button></div></div>
 <div><label for="cond">Target condition</label><select id="cond"></select></div>
 <div><label for="arm">Model / arm</label><select id="arm"></select></div>
 <div class="na" id="armnote" style="max-width:360px"></div>
</div>
<div id="err"></div>
<div class="panels" id="panels"></div>
<div class="grid2">
 <div class="card"><h2>Scores and image metrics <span class="na">(unavailable = not in the saved results)</span></h2><table id="metrics"></table>
  <p class="cav">p(...) comes from a frozen DenseNet121 trained on automatic CheXpert labels. B and C were optimised against it, so a higher score is <b>not</b> evidence that the condition was generated correctly or that the image is clinically valid. PSNR/SSIM compare the edit with the reconstruction of the input.</p></div>
 <div class="card"><h2>Context</h2><div id="ctx"></div></div>
</div>
<div class="card" style="margin-top:14px"><h2>All arms for this case</h2><div class="strip" id="strip"></div></div>
<footer>Counterfactuals are synthetic images produced by a research model, not real patient outcomes. Conclusion of the study: the hypothesis that the B/C constraints improve counterfactual reliability was not supported under the tested conditions (one seed, 24 held-out images, one edit operation). See docs/FINAL_RESEARCH_REPORT.md.</footer>
</main>
<script>
const $=id=>document.getElementById(id);let IDX=null,CUR=null;
function el(tag,attrs,...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs||{})){if(k==='class')e.className=v;else e.setAttribute(k,v)}for(const c of kids){e.append(c)}return e}
function fmt(m){if(m.value===null||m.value===undefined)return null;const v=m.value;switch(m.format){case'prob':return v.toFixed(2);case'signed':return(v>=0?'+':'−')+Math.abs(v).toFixed(2);case'db':return v.toFixed(1)+' dB';case'ssim':return v.toFixed(3);case'pct':return v.toFixed(1)+' %';default:return String(v)}}
async function api(u){const r=await fetch(u);const j=await r.json().catch(()=>({error:'bad response'}));if(!r.ok)throw new Error(j.error||r.statusText);return j}
function currentHash(){return `case=${$('case').value}&arm=${$('arm').value}&cond=${encodeURIComponent($('cond').value)}`}
function setHash(){const h=currentHash();if(location.hash.slice(1)!==h)location.hash=h}
function readHash(){const p=new URLSearchParams(location.hash.slice(1));return{case:p.get('case'),arm:p.get('arm'),cond:p.get('cond')}}
async function init(){try{IDX=await api('/api/index')}catch(e){$('err').textContent='Cannot load the saved results: '+e.message;return}
 if(IDX.problems.length)$('err').textContent='Note: '+IDX.problems.join('; ');
 const h=readHash();
 for(const c of IDX.cases)$('case').append(el('option',{value:c},'Case '+c));
 for(const c of IDX.conditions){const o=el('option',{value:c.key},c.label+(c.available?'':' — unavailable'));if(!c.available)o.disabled=true;$('cond').append(o)}
 const first=IDX.conditions.find(c=>c.available);if(first)$('cond').value=h.cond&&IDX.conditions.some(c=>c.key===h.cond&&c.available)?h.cond:first.key;
 if(h.case!==null&&IDX.cases.includes(Number(h.case)))$('case').value=h.case;
 await refresh(h.arm)}
async function refresh(prefArm){const c=$('case').value,k=$('cond').value;$('err').textContent=IDX.problems.length?'Note: '+IDX.problems.join('; '):'';
 try{CUR=await api(`/api/case?case=${encodeURIComponent(c)}&condition=${encodeURIComponent(k)}`)}catch(e){$('err').textContent=e.message;CUR=null;$('panels').replaceChildren();return}
 const prev=prefArm||$('arm').value;$('arm').replaceChildren();
 for(const[a,e]of Object.entries(CUR.arms)){const o=el('option',{value:a},e.title+(e.available?'':' — unavailable'));if(!e.available)o.disabled=true;$('arm').append(o)}
 const av=Object.keys(CUR.arms).filter(a=>CUR.arms[a].available);
 if(!av.length){$('panels').replaceChildren(el('div',{class:'card na'},'No saved output exists for this case and condition.'));$('metrics').replaceChildren();$('ctx').replaceChildren();$('strip').replaceChildren();return}
 $('arm').value=av.includes(prev)?prev:av[0];render();setHash()}
function panel(tag,tagc,title,src,extra){const f=el('figure');if(src){const i=el('img',{src,alt:title});f.append(i)}else f.append(el('div',{class:'missing'},'unavailable — not in the saved results'));
 f.append(el('figcaption',{},el('span',{class:'tag '+tagc},tag),title,extra?el('div',{class:'na'},extra):''));return f}
function render(){const arm=$('arm').value,e=CUR.arms[arm],P=$('panels');P.replaceChildren();
 $('armnote').textContent='Research finding for this arm (study report): '+e.finding;
 const im=e.images;
 P.append(panel('INPUT','t-in','Original X-ray',im.original,'real chest X-ray, center-cropped'));
 P.append(panel('SYNTHETIC','t-syn','Reconstruction',im.reconstruction,'inversion of the input — a fidelity check, not an edit'));
 P.append(panel('COUNTERFACTUAL','t-syn',`${e.title}: edited (${CUR.pathology}, add)`,im.counterfactual,'synthetic image — not a real patient outcome'));
 P.append(panel('CHANGE MAP','t-map','|edit − reconstruction| ×4',im.change,'brighter = larger change'));
 const T=$('metrics');T.replaceChildren();for(const m of Object.values(e.metrics)){const s=fmt(m);T.append(el('tr',{},el('td',{},m.label),el('td',{class:'v'+(s===null?' na':'')},s===null?'unavailable':s)))}
 const C=$('ctx');C.replaceChildren();const L=CUR.input.labels||{};
 C.append(el('p',{},`Case ${CUR.case} · held-out test patient · ${CUR.input.ap_pa||'view unavailable'}`));
 C.append(el('p',{},'Input labels (automatic labeler): '+(Object.keys(L).length?Object.entries(L).map(([k,v])=>`${k}=${v===null?'unmentioned':v}`).join(', '):'unavailable')));
 for(const n of CUR.notes)C.append(el('p',{class:'na'},n));
 C.append(el('p',{class:'cav'},'Only “add Cardiomegaly” was evaluated in this project; other findings/operations are unavailable. B and C weights are not available locally, so nothing here can be regenerated.'));
 const S=$('strip');S.replaceChildren();for(const[a,x]of Object.entries(CUR.arms)){const d=el('div',{class:'mini'+(x.available?'':' off')+(a===arm?' sel':'')});
  if(x.available&&x.images.counterfactual){d.append(el('img',{src:x.images.counterfactual,alt:a}))}else d.append(el('div',{class:'missing'},'unavailable'));
  const pe=x.metrics.p_edit?fmt(x.metrics.p_edit):null;d.append(el('div',{},x.title),el('div',{class:'na'},x.available?('classifier p after edit: '+(pe===null?'unavailable':pe)):(x.reason||'unavailable')));
  if(x.available)d.onclick=()=>{$('arm').value=a;render();setHash()};S.append(d)}}
function step(d){const o=[...$('case').options],i=$('case').selectedIndex+d;if(i>=0&&i<o.length){$('case').selectedIndex=i;refresh()}}
$('case').onchange=()=>refresh();$('cond').onchange=()=>refresh();$('arm').onchange=()=>{render();setHash()};$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
document.addEventListener('keydown',ev=>{if(ev.target.tagName==='SELECT')return;if(ev.key==='ArrowLeft')step(-1);if(ev.key==='ArrowRight')step(1)});
window.addEventListener('hashchange',async()=>{if(!IDX||location.hash.slice(1)===currentHash())return;const h=readHash();
 if(h.case!==null&&IDX.cases.includes(Number(h.case)))$('case').value=h.case;
 if(h.cond&&IDX.conditions.some(c=>c.key===h.cond&&c.available))$('cond').value=h.cond;await refresh(h.arm)});
init();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
