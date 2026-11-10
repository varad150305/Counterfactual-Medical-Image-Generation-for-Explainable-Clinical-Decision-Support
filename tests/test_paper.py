"""Static checks of the paper source (paper/main.tex). No LaTeX toolchain is assumed (none is installed here), so these tests cannot prove that the PDF compiles;
they check what can be checked without it: balanced environments/braces, resolvable citations/references/inputs/figures, up-to-date generated tables,
the numbers quoted in the text against docs/results/results.json, the required honesty statements, and that no patient-level data is part of the paper folder."""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"


def tex() -> str:
    return (PAPER / "main.tex").read_text(encoding="utf-8")


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def all_tex() -> str:
    parts = [strip_comments(tex())]
    for f in sorted((PAPER / "generated").glob("*.tex")):
        parts.append(strip_comments(f.read_text(encoding="utf-8")))
    return "\n".join(parts)


def test_environments_and_braces_are_balanced():
    for name, text in [("main", strip_comments(tex()))] + [(f.name, strip_comments(f.read_text(encoding="utf-8"))) for f in (PAPER / "generated").glob("*.tex")]:
        begins, ends = Counter(re.findall(r"\\begin\{([^}]+)\}", text)), Counter(re.findall(r"\\end\{([^}]+)\}", text))
        assert begins == ends, f"{name}: unbalanced environments {begins - ends} / {ends - begins}"
        flat = re.sub(r"\\[{}]", "", text)
        assert flat.count("{") == flat.count("}"), f"{name}: unbalanced braces"


def test_citations_references_inputs_and_figures_resolve():
    text = all_tex()
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", text) for k in grp.split(",")}
    assert cited <= bib, f"undefined citations: {cited - bib}"
    assert bib <= cited, f"uncited bibliography entries: {bib - cited}"
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
    assert refs <= labels, f"undefined references: {refs - labels}"
    for f in re.findall(r"\\input\{([^}]+)\}", strip_comments(tex())):
        assert (PAPER / f).exists(), f
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(tex())):
        if f != "figures/qualitative.png":                       # optional figure, included only if the file exists (CheXpert-derived: not shipped)
            assert (PAPER / f).exists(), f
    assert not (PAPER / "figures" / "qualitative.png").exists(), "CheXpert-derived image must not live in the repository"
    assert (PAPER / "figures" / "qualitative.png").exists() is False


def test_generated_tables_are_up_to_date():
    spec = importlib.util.spec_from_file_location("paper_make_tables", PAPER / "make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, text in mod.build().items():
        assert (PAPER / "generated" / name).read_text(encoding="utf-8") == text, f"{name} is stale: run python paper/make_tables.py"


def test_numbers_quoted_in_the_text_match_the_results_file():
    r = json.loads((REPO / "docs" / "results" / "results.json").read_text())
    p, s, sat = r["paired"], r["summary"], r["saturation"]
    t = tex()
    expected = {
        "B-A_ft dp": f"{p['B-A_ft']['dp_target']['mean_diff']:.2f}", "B-A_ft dp ci lo": f"{p['B-A_ft']['dp_target']['ci95'][0]:.2f}", "B-A_ft dp ci hi": f"{p['B-A_ft']['dp_target']['ci95'][1]:.2f}",
        "B-A_ft psnr": f"{abs(p['B-A_ft']['psnr_recon_cf']['mean_diff']):.1f}", "C-B psnr": f"{p['C-B']['psnr_recon_cf']['mean_diff']:.1f}",
        "C-A_ft psnr": f"{abs(p['C-A_ft']['psnr_recon_cf']['mean_diff']):.1f}", "A_ft-A0 dp": f"{abs(p['A_ft-A0']['dp_target']['mean_diff']):.3f}",
        "B pixels": f"{round(100 * s['B']['median']['change_frac_changed_gt20'])}", "pooled rho": f"{abs(sat['pooled']['spearman_dp_vs_psnr']['rho']):.2f}",
        "pooled n>=0.9": str(sat["pooled"]["n_p_ge_0.9"]), "pooled n>=0.9&psnr<22": str(sat["pooled"]["n_p_ge_0.9_and_psnr_lt_22"]),
        "C collapsed": str(s["C"]["psnr_below_20dB"]), "C >15%": str(s["C"]["pixels_changed_over_15pct"]),
        "clip C": f"{100 * r['compute']['training']['C']['clip_fraction']:.1f}", "B p after": f"{s['B']['median']['p_target_cf']:.2f}",
    }
    for key, val in expected.items():
        assert val in t, f"number for '{key}' ({val}) not found in main.tex: text and results.json disagree"
    assert r["n_images"] == 24 and "24 held-out" in t.replace("\n", " ")


def test_honesty_statements_and_template_are_present():
    t = tex()
    assert r"\documentclass[conference]{IEEEtran}" in t
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1)
    for needle in ("not supported", "one seed", "24 images", "unreliable evidence"):
        assert needle in abstract, needle
    for sec in ("Limitations and Threats to Validity", "Reproducibility and Ethics", "Conclusion"):
        assert f"\\section{{{sec}}}" in t or f"\\section*{{{sec}}}" in t, sec
    low = t.lower()
    for needle in ("not clinically validated", "no independent classifier", "multiplicity", "not redistributed", "one non-clinical reviewer", "reconstruction of pris"):
        assert needle in low, needle
    conclusion = low.split("\\section{conclusion}")[1].split("\\section*")[0]
    assert "was not supported" in conclusion and "unreliable proxy" in conclusion


def test_no_patient_level_data_or_kaggle_paths_in_the_paper_folder():
    for f in PAPER.rglob("*"):
        if f.is_file() and f.suffix in {".tex", ".md", ".py"}:
            txt = f.read_text(encoding="utf-8")
            assert not re.search(r"patient\d{4,}", txt) and "/kaggle/" not in txt, f
    assert sorted(p.name for p in (PAPER / "figures").iterdir()) == ["fig1_disease_response.png", "fig2_preservation.png", "fig4_tradeoff.png", "fig5_paired_differences.png"]


def test_paper_readme_lists_the_placeholders_and_the_build_limitation():
    text = (PAPER / "README.md").read_text(encoding="utf-8")
    for needle in ("[Department / Institution]", "not been compiled", "make_tables.py", "pdflatex", "qualitative.png", "references"):
        assert needle in text, needle
    assert "[Department / Institution]" in tex()


def test_bibliography_is_numbered_in_order_of_first_citation():
    """IEEE numbers references in the order they are first cited; the inline bibliography must follow that order."""
    t = strip_comments(tex())
    head = t.split("\\begin{thebibliography}")[0]
    order = []
    for grp in re.findall(r"\\cite\{([^}]+)\}", head):
        for k in (k.strip() for k in grp.split(",")):
            if k not in order:
                order.append(k)
    assert re.findall(r"\\bibitem\{([^}]+)\}", t) == order


def test_word_content_model_converts_the_whole_paper():
    """tex2model.py (source of paper.docx) must parse main.tex without dropping or failing on any construct."""
    spec = importlib.util.spec_from_file_location("paper_tex2model", PAPER / "tex2model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blocks = mod.convert()["blocks"]
    kinds = Counter(b["type"] for b in blocks)
    assert kinds["table"] == 5 and kinds["figure"] == 4 and kinds["equation"] == 3
    assert len(next(b for b in blocks if b["type"] == "refs")["items"]) == len(re.findall(r"\\bibitem\{", tex()))
    text = " ".join("".join(r["t"] for r in b.get("runs", [])) for b in blocks)
    assert "\\" not in text and "$" not in text
