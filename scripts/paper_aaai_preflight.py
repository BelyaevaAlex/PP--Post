#!/usr/bin/env python3
"""AAAI paper preflight for the PP-Post manuscript.

The script is intentionally conservative: it can rebuild PDFs when a local TeX
engine exists, but it also produces a useful structural report when TeX is not
installed in the current environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "aaai_pppost_mortality"
GENERATED_DIR = PAPER_DIR / "generated"

INTERNAL_PATTERNS = [
    "pp_theta_post",
    "tabpfn_distill",
    "source_native",
    "ebm_terms",
    "mask(",
    "value(",
    "rahmatullaev",
    "cache",
    "bounded_residual",
    "residual_mcc",
    "rule_family",
    "best_mcc",
    "negative_boundary",
    "section_",
    "job",
    "mlspace",
]


@dataclass
class PdfStatus:
    name: str
    exists: bool
    pages: int | None
    stale: bool | None
    pdf_mtime: str | None
    newest_source_mtime: str | None
    references_page: int | None


@dataclass
class TexStatus:
    name: str
    line_count: int
    approx_words: int
    inputs: list[str]
    tables: int
    wide_tables: int
    figures: int
    wide_figures: int
    sections: int
    subsections: int


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_latex_for_words(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\\(?:cite|ref|label|input|includegraphics)(?:\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$^_&]", " ", text)
    return text


def resolve_inputs(tex_path: Path, seen: set[Path] | None = None) -> list[Path]:
    seen = seen or set()
    if tex_path in seen or not tex_path.exists():
        return []
    seen.add(tex_path)
    text = read_text(tex_path)
    found: list[Path] = []
    for raw in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = (tex_path.parent / raw).resolve()
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".tex")
        found.append(candidate)
        found.extend(resolve_inputs(candidate, seen))
    return found


def tex_status(tex_path: Path) -> TexStatus:
    text = read_text(tex_path)
    inputs = resolve_inputs(tex_path)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", strip_latex_for_words(text))
    return TexStatus(
        name=tex_path.name,
        line_count=len(text.splitlines()),
        approx_words=len(words),
        inputs=[str(p.relative_to(PAPER_DIR)) if p.is_relative_to(PAPER_DIR) else str(p) for p in inputs],
        tables=len(re.findall(r"\\begin\{table\}", text)),
        wide_tables=len(re.findall(r"\\begin\{table\*\}", text)),
        figures=len(re.findall(r"\\begin\{figure\}", text)),
        wide_figures=len(re.findall(r"\\begin\{figure\*\}", text)),
        sections=len(re.findall(r"\\section\{", text)),
        subsections=len(re.findall(r"\\subsection\{", text)),
    )


def newest_source_mtime(paths: Iterable[Path]) -> float | None:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else None


def fmt_mtime(ts: float | None) -> str | None:
    if ts is None:
        return None
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def pdf_pages(pdf_path: Path) -> int | None:
    if not pdf_path.exists() or shutil.which("pdfinfo") is None:
        return None
    proc = run(["pdfinfo", str(pdf_path)], cwd=PAPER_DIR)
    match = re.search(r"^Pages:\s+(\d+)", proc.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def references_page(pdf_path: Path) -> int | None:
    if not pdf_path.exists() or shutil.which("pdftotext") is None:
        return None
    proc = run(["pdftotext", "-layout", str(pdf_path), "-"], cwd=PAPER_DIR)
    if proc.returncode != 0:
        return None
    pages = proc.stdout.split("\f")
    for idx, page in enumerate(pages, start=1):
        if re.search(r"(?m)^\s*References\s*$", page):
            return idx
    return None


def pdf_status(tex_path: Path) -> PdfStatus:
    pdf_path = tex_path.with_suffix(".pdf")
    source_paths = [tex_path, PAPER_DIR / "references.bib", PAPER_DIR / "aaai2027.sty", PAPER_DIR / "aaai2027.bst"]
    source_paths.extend(resolve_inputs(tex_path))
    newest = newest_source_mtime(source_paths)
    pdf_mtime = pdf_path.stat().st_mtime if pdf_path.exists() else None
    stale = None
    if pdf_mtime is not None and newest is not None:
        stale = pdf_mtime < newest
    return PdfStatus(
        name=pdf_path.name,
        exists=pdf_path.exists(),
        pages=pdf_pages(pdf_path),
        stale=stale,
        pdf_mtime=fmt_mtime(pdf_mtime),
        newest_source_mtime=fmt_mtime(newest),
        references_page=references_page(pdf_path),
    )


def build_pdf(tex_name: str) -> dict[str, object]:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if latexmk:
        proc = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_name], cwd=PAPER_DIR)
        return {"tool": "latexmk", "returncode": proc.returncode, "output_tail": proc.stdout[-4000:]}
    if pdflatex:
        steps = [
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_name],
        ]
        aux_name = Path(tex_name).stem
        if bibtex:
            steps.append([bibtex, aux_name])
        steps.extend(
            [
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_name],
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_name],
            ]
        )
        outputs: list[str] = []
        rc = 0
        for step in steps:
            proc = run(step, cwd=PAPER_DIR)
            outputs.append(proc.stdout[-2000:])
            rc = proc.returncode
            if rc != 0:
                break
        return {"tool": "pdflatex", "returncode": rc, "output_tail": "\n".join(outputs)[-4000:]}
    return {
        "tool": None,
        "returncode": None,
        "output_tail": "No latexmk or pdflatex found in PATH; PDF rebuild skipped.",
    }


def log_findings(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    out = []
    for line in read_text(log_path).splitlines():
        if re.search(r"Overfull|Float too large|LaTeX Warning|undefined|Citation|Error", line):
            out.append(line.strip())
    return out


def scan_internal_labels(paths: Iterable[Path]) -> list[dict[str, object]]:
    findings = []
    for path in paths:
        if not path.exists() or path.suffix not in {".tex", ".md", ".csv"}:
            continue
        text = read_text(path)
        for pattern in INTERNAL_PATTERNS:
            if pattern in text:
                lines = [idx for idx, line in enumerate(text.splitlines(), start=1) if pattern in line]
                findings.append(
                    {
                        "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                        "pattern": pattern,
                        "lines": lines[:10],
                        "count": len(lines),
                    }
                )
    return findings


def render_markdown(report: dict[str, object]) -> str:
    main_pdf = next(p for p in report["pdfs"] if p["name"] == "main.pdf")
    supp_pdf = next(p for p in report["pdfs"] if p["name"] == "supplementary.pdf")
    status = []
    if not main_pdf["exists"]:
        status.append("main PDF is missing")
    elif main_pdf["stale"]:
        status.append("main PDF is stale")
    elif main_pdf["pages"] is not None and main_pdf["pages"] > 9:
        status.append("main PDF exceeds 9 total pages")
    elif main_pdf["pages"] is not None and main_pdf["pages"] <= 9:
        status.append("main PDF page count is within total-page cap")
    if supp_pdf["stale"]:
        status.append("supplement PDF is stale")
    if report["internal_label_findings"]:
        status.append("internal labels remain in included paper artifacts")
    if not status:
        status.append("no blocking structural issue detected")

    lines = [
        "# AAAI Submission Preflight",
        "",
        "## Verdict",
        "",
        *[f"- {item}" for item in status],
        "",
        "## PDF Status",
        "",
        "| PDF | exists | pages | stale | PDF mtime | newest source mtime | References page |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for pdf in report["pdfs"]:
        lines.append(
            "| {name} | {exists} | {pages} | {stale} | {pdf_mtime} | {newest_source_mtime} | {references_page} |".format(
                **{k: ("" if v is None else v) for k, v in pdf.items()}
            )
        )
    lines.extend(["", "## TeX Structure", ""])
    lines.append("| File | lines | approx words | inputs | tables | wide tables | figures | sections/subsections |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tex in report["tex"]:
        lines.append(
            f"| {tex['name']} | {tex['line_count']} | {tex['approx_words']} | "
            f"{len(tex['inputs'])} | {tex['tables']} | {tex['wide_tables']} | "
            f"{tex['figures'] + tex['wide_figures']} | {tex['sections']}/{tex['subsections']} |"
        )
    lines.extend(["", "## Build Attempts", ""])
    for build in report["builds"]:
        lines.append(f"- `{build['tex']}`: tool={build['tool']}, returncode={build['returncode']}")
        if build.get("output_tail"):
            lines.append("")
            lines.append("```")
            lines.append(str(build["output_tail"]).strip())
            lines.append("```")
            lines.append("")
    lines.extend(["", "## LaTeX Log Findings", ""])
    for name, findings in report["logs"].items():
        lines.append(f"- `{name}`: {len(findings)} findings")
        for item in findings[:20]:
            lines.append(f"  - {item}")
    if report["internal_label_findings"]:
        lines.extend(["", "## Internal Label Findings", ""])
        for item in report["internal_label_findings"]:
            lines.append(f"- `{item['file']}`: `{item['pattern']}` at lines {item['lines']} (count={item['count']})")
    else:
        lines.extend(["", "## Internal Label Findings", "", "- None in included TeX/generated artifacts."])
    lines.extend(
        [
            "",
            "## Submission Policy Reminder",
            "",
            "- AAAI-27 main-track papers should fit the official author-kit style.",
            "- The main paper must stay self-contained; supplementary material may support claims but should not carry the core contribution.",
            "- Keep the main technical content to the official page budget; pages after the content limit should contain only references or permitted checklist material.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    global PAPER_DIR, GENERATED_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--build", action="store_true", help="Attempt to rebuild main and supplementary PDFs.")
    parser.add_argument("--json", type=Path, default=GENERATED_DIR / "aaai_preflight_report.json")
    parser.add_argument("--md", type=Path, default=GENERATED_DIR / "aaai_preflight_report.md")
    args = parser.parse_args(argv)

    PAPER_DIR = args.paper_dir.resolve()
    GENERATED_DIR = PAPER_DIR / "generated"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    builds = []
    if args.build:
        for tex in ["main.tex", "supplementary.tex"]:
            result = build_pdf(tex)
            result["tex"] = tex
            builds.append(result)
    else:
        builds.append({"tex": "main.tex/supplementary.tex", "tool": None, "returncode": None, "output_tail": "Build not requested."})

    tex_paths = [PAPER_DIR / "main.tex", PAPER_DIR / "supplementary.tex"]
    tex_reports = [asdict(tex_status(path)) for path in tex_paths]
    pdf_reports = [asdict(pdf_status(path)) for path in tex_paths]
    included = set(tex_paths)
    for path in tex_paths:
        included.update(resolve_inputs(path))
    report = {
        "paper_dir": str(PAPER_DIR),
        "builds": builds,
        "tex": tex_reports,
        "pdfs": pdf_reports,
        "logs": {
            "main.log": log_findings(PAPER_DIR / "main.log"),
            "supplementary.log": log_findings(PAPER_DIR / "supplementary.log"),
        },
        "internal_label_findings": scan_internal_labels(sorted(included)),
        "tool_paths": {
            "latexmk": shutil.which("latexmk"),
            "pdflatex": shutil.which("pdflatex"),
            "bibtex": shutil.which("bibtex"),
            "pdfinfo": shutil.which("pdfinfo"),
            "pdftotext": shutil.which("pdftotext"),
        },
    }
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.md}")
    print(f"Wrote {args.json}")

    blocking = []
    main_pdf = next(p for p in pdf_reports if p["name"] == "main.pdf")
    if main_pdf["stale"]:
        blocking.append("main PDF is stale")
    if main_pdf["pages"] is not None and main_pdf["pages"] > 9:
        blocking.append("main PDF exceeds 9 pages")
    if report["internal_label_findings"]:
        blocking.append("internal labels found")
    if blocking:
        print("Blocking:", "; ".join(blocking))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
