"""Proposal completion (2026-10-22): collect the generated summaries into the committed tables file
reports/completion_tables.md (every number there comes from these generated files; nothing is typed by hand).

Sections (each the summary.md of one script, run first):
  1  classification_conditions.py   precision / recall / F1 / accuracy, results per recording condition
  2  temporal_stability.py          jitter, jitter variance, smoothness; rep-to-rep stability of the assessment
  3  depth_check.py                 is the CVDC embedding depth-aware (side camera as the depth reference)
  4  module3_comparison.py          Module III with the CVDC representation (M19) and angular velocity (M20), 5-fold
  5  module3_comparison.py --loso   the same under LOSO
  6  depth_baseline_report.py       B2-DepthAnything next to the other networks (5-fold and LOSO)

usage: python scripts/full/completion/report_completion.py
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402

PARTS = [("quick", "1. Classification metrics and recording conditions"),
         ("temporal", "2. Temporal stability"),
         ("depth", "3. Is the CVDC embedding depth-aware?"),
         ("module3_comparison", "4. Module III with the CVDC representation and angular velocity (GroupKFold-5)"),
         ("module3_comparison_loso", "5. Module III with the CVDC representation and angular velocity (LOSO)"),
         ("depth_baseline", "6. B2-DepthAnything (baseline 2 with DepthAnything depth)")]


def main():
    cfg = load_config(REPO / "configs" / "completion.yaml")
    base = resolve(cfg["eval"]["results_dir"]) / "completion"
    L = ["# Proposal completion: generated tables (E01–E04)\n",
         f"Assembled by `scripts/full/completion/report_completion.py` on {time.strftime('%Y-%m-%d %H:%M')} from the "
         f"summaries in `outputs/full/results/completion/`. Narrative and decisions: "
         f"[`FINAL_REPORT_E01_E04.md`](FINAL_REPORT_E01_E04.md) §4–§7, `DESIGN_DECISIONS.md` D19, M19, M20, A11, E31, "
         f"E32, X11. The pipeline's output (Module III's score) has its own tables: "
         f"[`pipeline_tables.md`](pipeline_tables.md).\n"]
    for folder, title in PARTS:
        p = base / folder / "summary.md"
        L.append(f"\n---\n\n# {title}\n")
        if not p.exists():
            L.append(f"_Not generated yet (`outputs/full/results/completion/{folder}/summary.md` is missing)._\n")
            continue
        body = p.read_text(encoding="utf-8").splitlines()
        if body and body[0].startswith("# "):
            body = body[1:]
        L += ["#" + x if x.startswith("#") else x for x in body]  # demote headings one level
    out = REPO / "reports" / "completion_tables.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
