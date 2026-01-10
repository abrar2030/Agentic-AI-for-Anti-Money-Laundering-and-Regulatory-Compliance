"""Populate a LaTeX-friendly results snippet from experiment output.

This small utility extracts key metrics from `full_experiments.json` and
writes a simple LaTeX table fragment that can be `\\input{}` into a paper
or report.
"""

import json
from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)


def main(
    results_dir: str = "results/full_experiments",
    output_file: str = "paper_results.tex",
):
    results_path = Path(results_dir) / "full_experiments.json"
    if not results_path.exists():
        raise RuntimeError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    baseline = results.get("baseline_results", {}).get("xgboost", {})
    agentic = results.get("agentic_results", {})

    tex = []
    tex.append("% Auto-generated results snippet")
    tex.append("\\begin{table}[ht]")
    tex.append("\\centering")
    tex.append("\\begin{tabular}{lccc}")
    tex.append("\\toprule")
    tex.append("Model & Precision & Recall & F1 \\")
    tex.append("\\midrule")
    tex.append(
        f"XGBoost & {baseline.get('precision', 0.0):.3f} & {baseline.get('recall', 0.0):.3f} & {baseline.get('f1', 0.0):.3f} \\"
    )
    tex.append(
        f"Agentic & {agentic.get('precision', 0.0):.3f} & {agentic.get('recall', 0.0):.3f} & {agentic.get('f1', 0.0):.3f} \\"
    )
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Key model comparison (auto-generated).}")
    tex.append("\\end{table}")

    out_path = Path(output_file)
    out_path.write_text("\n".join(tex))

    logger.info("LaTeX snippet written to: %s", out_path)
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/full_experiments")
    parser.add_argument("--output-file", type=str, default="paper_results.tex")

    args = parser.parse_args()
    main(results_dir=args.results_dir, output_file=args.output_file)
