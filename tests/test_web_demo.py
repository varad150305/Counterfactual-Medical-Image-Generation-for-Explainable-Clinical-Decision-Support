"""Tests of the localhost web viewer (app/web_demo.py): artifact discovery, case/arm/condition selection, missing outputs, metric preparation,
HTTP behaviour and the safety notice. Uses the synthetic Phase 6/7 artifact tree of the Phase 8 tests (real layout, no patient data)."""
import importlib.util
import json
import re
import shutil
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from test_phase8_report import clf_ckpt, tree  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]
REQUIRED_NOTICE = ("Research demonstration using synthetic counterfactual medical images. Not clinically validated. "
                   "Not for diagnosis or medical decision-making. Classifier scores do not establish clinical correctness.")


def load_web():
    spec = importlib.util.spec_from_file_location("p_web_demo", REPO / "app" / "web_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def web():
    return load_web()


def copy_roots(tree, dest):
    shutil.copytree(tree["roots"]["phase6"], dest / "p6")
    shutil.copytree(tree["roots"]["phase7"], dest / "p7")
    return dest / "p6", dest / "p7"


@contextmanager
def running(web, catalog):
    server = web.make_server(catalog, "127.0.0.1", 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def jget(url):
    code, _, body = get(url)
    return code, json.loads(body)


class TestDiscovery:
    def test_cases_conditions_and_arms_come_from_the_artifacts(self, web, tree):
        cat = web.Catalog(tree["roots"]["phase6"], tree["roots"]["phase7"])
        idx = cat.index_payload()
        assert idx["cases"] == list(range(6)) and idx["notice"] == REQUIRED_NOTICE
        available = [c for c in idx["conditions"] if c["available"]]
        assert [c["key"] for c in available] == ["Cardiomegaly|add"]                      # the only condition represented
        unavailable = {c["label"]: c["reason"] for c in idx["conditions"] if not c["available"]}
        assert set(unavailable) == {"Pneumonia (add)", "Pleural Effusion (add)", "No Finding (add)"} and all("not represented" in r for r in unavailable.values())
        assert [a["key"] for a in idx["arms"]] == ["A0", "A_ft", "B", "C"] and all(a["available"] and a["cases"] == 6 for a in idx["arms"])

    def test_case_payload_has_images_and_metrics_from_the_stored_scores(self, web, tree):
        cat = web.Catalog(tree["roots"]["phase6"], tree["roots"]["phase7"])
        p = cat.case_payload(2, "Cardiomegaly|add")
        assert p["pathology"] == "Cardiomegaly" and set(p["arms"]) == {"A0", "A_ft", "B", "C"}
        c = p["arms"]["C"]
        assert c["available"] and c["images"] == {"original": "/img/C/2/original", "reconstruction": "/img/C/2/reconstruction",
                                                  "counterfactual": "/img/C/2/counterfactual", "change": "/img/C/2/change"}
        stored = pd.read_csv(tree["roots"]["phase7"] / "phase6" / "scores" / "per_image_C.csv").set_index("index").loc[2]
        m = c["metrics"]
        assert m["p_edit"]["value"] == pytest.approx(stored["p_target_cf"]) and m["psnr_edit"]["value"] == pytest.approx(stored["psnr_recon_cf"])
        assert m["changed"]["value"] == pytest.approx(100 * stored["change_frac_changed_gt20"]) and m["ssim_edit"]["value"] == pytest.approx(stored["ssim_recon_cf"])
        assert m["p_edit"]["label"] == "p(Cardiomegaly) after the edit" and "evidence" not in json.dumps(m)

    def test_missing_metric_columns_and_nan_are_none_not_zero(self, web, tree, tmp_path):
        p6, p7 = copy_roots(tree, tmp_path)
        csv = p7 / "phase6" / "scores" / "per_image_C.csv"
        df = pd.read_csv(csv)
        df = df.drop(columns=["ssim_orig_cf"])
        df.loc[df["index"] == 1, "clip_directional"] = float("nan")
        df.to_csv(csv, index=False)
        m = web.Catalog(p6, p7).case_payload(1, "Cardiomegaly|add")["arms"]["C"]["metrics"]
        assert m["ssim_edit_orig"]["value"] is None and m["clip"]["value"] is None            # unavailable, never 0
        assert m["psnr_edit"]["value"] is not None

    def test_arm_without_a_scores_file_still_shows_images_with_unavailable_metrics(self, web, tree, tmp_path):
        p6, p7 = copy_roots(tree, tmp_path)
        (p6 / "phase6" / "scores" / "per_image_B.csv").unlink()
        b = web.Catalog(p6, p7).case_payload(0, "Cardiomegaly|add")["arms"]["B"]
        assert b["available"] and b["images"]["counterfactual"] and all(v["value"] is None for v in b["metrics"].values())

    def test_missing_outputs_make_only_that_arm_and_case_unavailable(self, web, tree, tmp_path):
        p6, p7 = copy_roots(tree, tmp_path)
        next((p6 / "phase6_eval_B").rglob("03_add_Cardiomegaly")).joinpath("counterfactual.png").unlink()
        cat = web.Catalog(p6, p7)
        p3 = cat.case_payload(3, "Cardiomegaly|add")["arms"]
        assert p3["B"]["available"] is False and "incomplete" in p3["B"]["reason"] and all(p3[a]["available"] for a in ("A0", "A_ft", "C"))
        assert cat.case_payload(4, "Cardiomegaly|add")["arms"]["B"]["available"] is True
        shutil.rmtree(next((p6 / "phase6_eval_A0").iterdir()) / "outputs" / "01_add_Cardiomegaly")
        assert "no saved output" in web.Catalog(p6, p7).case_payload(1, "Cardiomegaly|add")["arms"]["A0"]["reason"]

    def test_a_missing_arm_directory_or_root_is_reported_and_the_rest_still_works(self, web, tree, tmp_path):
        p6, p7 = copy_roots(tree, tmp_path)
        shutil.rmtree(p7 / "phase7_eval_C")
        cat = web.Catalog(p6, p7)
        c = cat.case_payload(0, "Cardiomegaly|add")["arms"]
        assert c["C"]["available"] is False and "not found" in c["C"]["reason"] and c["A0"]["available"]
        cat2 = web.Catalog(p6, tmp_path / "nowhere")
        assert [a["key"] for a in cat2.index_payload()["arms"] if a["available"]] == ["A0", "A_ft", "B"] and cat2.problems
        with pytest.raises(web.WebDemoError, match="no saved results found"):
            web.Catalog(tmp_path / "x", tmp_path / "y")
        assert web.main(["--phase6-op", str(tmp_path / "x"), "--phase7-op", str(tmp_path / "y")]) == 2

    def test_a_condition_in_the_metadata_is_discovered_and_not_assumed_for_other_arms(self, web, tree, tmp_path):
        p6, p7 = copy_roots(tree, tmp_path)
        meta = next((p6 / "phase6_eval_A0").rglob("01_add_Cardiomegaly")) / "metadata.json"
        m = json.loads(meta.read_text())
        m.update({"target_pathology": "Pneumonia", "operation": "add"})
        meta.write_text(json.dumps(m))
        cat = web.Catalog(p6, p7)
        assert cat.conditions == ["Cardiomegaly|add", "Pneumonia|add"]
        assert cat.case_payload(1, "Cardiomegaly|add")["arms"]["A0"]["available"] is False
        assert "not represented" in cat.case_payload(1, "Cardiomegaly|add")["arms"]["A0"]["reason"]
        pn = cat.case_payload(1, "Pneumonia|add")["arms"]
        assert pn["A0"]["available"] and not any(pn[a]["available"] for a in ("A_ft", "B", "C"))
        assert {c["label"]: c["available"] for c in cat.index_payload()["conditions"]}["Pneumonia (add)"] is True

    def test_invalid_selections_raise(self, web, tree):
        cat = web.Catalog(tree["roots"]["phase6"], tree["roots"]["phase7"])
        with pytest.raises(web.WebDemoError, match="unknown case"):
            cat.case_payload(99, "Cardiomegaly|add")
        with pytest.raises(web.WebDemoError, match="not represented"):
            cat.case_payload(0, "Pneumonia|add")


class TestHttp:
    def test_page_notice_images_and_json(self, web, tree):
        cat = web.Catalog(tree["roots"]["phase6"], tree["roots"]["phase7"])
        with running(web, cat) as base:
            code, ctype, body = get(base + "/")
            html = body.decode("utf-8")
            assert code == 200 and "text/html" in ctype
            assert re.search(r'<div id="notice"[^>]*>' + re.escape(REQUIRED_NOTICE) + "</div>", html)                 # persistent, static, verbatim
            assert "position:sticky" in html and "COUNTERFACTUAL" in html and "INPUT" in html and "SYNTHETIC" in html
            assert not re.search(r"""(src|href)=["']?https?://""", html) and "cdn" not in html.lower() and "fetch('http" not in html     # no external resources
            code, idx = jget(base + "/api/index")
            assert code == 200 and idx["notice"] == REQUIRED_NOTICE and len(idx["cases"]) == 6
            code, case = jget(base + "/api/case?case=3&condition=Cardiomegaly%7Cadd")
            assert code == 200 and case["case"] == 3 and case["arms"]["B"]["available"]
            code, ctype, png = get(base + case["arms"]["B"]["images"]["counterfactual"])
            assert code == 200 and ctype == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n"
            code, ctype, png = get(base + case["arms"]["B"]["images"]["change"])
            assert code == 200 and ctype == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n"
            default_cond = jget(base + "/api/case?case=3")                       # condition defaults to the first represented one
            assert default_cond[0] == 200 and default_cond[1]["condition"] == "Cardiomegaly|add"

    def test_invalid_requests_are_rejected_without_crashing(self, web, tree):
        cat = web.Catalog(tree["roots"]["phase6"], tree["roots"]["phase7"])
        with running(web, cat) as base:
            assert jget(base + "/api/case?case=abc")[0] == 400
            assert jget(base + "/api/case?case=99")[0] == 404
            assert jget(base + "/api/case?case=1&condition=Pneumonia%7Cadd")[0] == 404
            assert get(base + "/img/Z/1/original")[0] == 404 and get(base + "/img/A0/99/original")[0] == 404
            assert get(base + "/img/A0/1/secret")[0] == 404 and get(base + "/img/A0/x/original")[0] == 400
            assert get(base + "/img/..%2F..%2Fp7/1/original")[0] == 404 and get(base + "/img/A0/..%5C..%5C/original")[0] in (400, 404)
            assert get(base + "/api/nothing")[0] == 404 and get(base + "/../../etc/passwd")[0] == 404
            assert jget(base + "/api/index")[0] == 200                                # still alive

    def test_a_missing_image_file_is_a_404_not_a_crash(self, web, tree, tmp_path):
        p6, p7 = copy_roots(tree, tmp_path)
        cat = web.Catalog(p6, p7)
        next((p6 / "phase6_eval_A_ft").rglob("02_add_Cardiomegaly")).joinpath("reconstructed.png").unlink()
        with running(web, cat) as base:
            assert get(base + "/img/A_ft/2/reconstruction")[0] == 404 and get(base + "/img/A_ft/2/change")[0] == 404
            case = jget(base + "/api/case?case=2")[1]["arms"]["A_ft"]
            assert case["available"] and case["images"]["reconstruction"] is None and case["images"]["change"] is None       # shown as unavailable

    def test_server_only_binds_to_loopback(self, web, tree):
        cat = web.Catalog(tree["roots"]["phase6"], tree["roots"]["phase7"])
        with pytest.raises(web.WebDemoError, match="loopback"):
            web.make_server(cat, "0.0.0.0", 0)


class TestSafetyAndDocs:
    def test_notice_text_is_the_required_one_and_findings_are_not_framed_as_success(self, web):
        assert web.NOTICE == REQUIRED_NOTICE
        page = web.PAGE
        assert "not</b> evidence" in page and "not a real patient outcome" in page
        assert "cannot regenerate anything" in page and "B and C weights are not available locally" in page
        assert "hashchange" in page and "location.hash" in page          # shareable / editable links (regression: the hash was ignored after load)

    def test_readme_documents_the_web_demo_command(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        assert "python app/web_demo.py" in text or "app/web_demo.py --phase6-op" in text
        assert "http://127.0.0.1:8765" in text

    def test_the_cli_demo_is_untouched_and_still_importable(self):
        spec = importlib.util.spec_from_file_location("p_cli_demo", REPO / "app" / "demo.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert m.DISCLAIMER.startswith("SYNTHETIC COUNTERFACTUAL IMAGES") and set(m.ARMS) == {"A0", "A_ft", "B", "C"}
