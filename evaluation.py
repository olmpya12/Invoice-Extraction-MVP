#!/usr/bin/env python3
"""Enhanced evaluator that produces on-disk reports for invoice-extraction runs.

Usage (stand-alone):
    python evaluation.py \
        --ground-truths ground_truths \
        --predictions outputs/regex \
        --out-dir     reports/regex_eval

It can still be imported from *main.py*:

    from evaluation import InvoiceEvaluator
    evaluator = InvoiceEvaluator(Path("ground_truths"), Path(pred_dir))
    evaluator.evaluate()
    summary, details = evaluator.report()  # report() also saves CSV+JSON

Directory structure created:
    <out-dir>/summary.json        – overall accuracies
    <out-dir>/details.csv         – per-invoice metrics & mismatches
    <out-dir>/details.json        – same, JSON-serialised
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _values_close(a: float, b: float, tol: float = 1e-2) -> bool:
    """Return True if two floats are within *tol*."""
    return abs(a - b) <= tol


# ────────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ────────────────────────────────────────────────────────────────────────────────

class InvoiceEvaluator:
    """Compare prediction JSONs against ground-truths and create full reports."""

    #: Columns to appear in the detailed CSV/JSON report
    _CSV_COLUMNS: Tuple[str, ...] = (
        "file",
        "po_match",
        "po_gt",
        "po_pred",
        "line_items_correct",
        "line_items_total",
        "line_item_accuracy",
        "totals_match",
        "subtotal_gt",
        "subtotal_pred",
        "vat_gt",
        "vat_pred",
        "total_gt",
        "total_pred",
    )

    def __init__(
        self,
        ground_truth_dir: Path,
        prediction_dir: Path,
        output_dir: Path | None = None,
        tol: float = 1e-2,
    ) -> None:
        self.gt_dir = ground_truth_dir
        self.pred_dir = prediction_dir
        self.tol = tol
        self.results: List[Dict[str, Any]] = []
        self.output_dir = (
            output_dir if output_dir is not None else prediction_dir / "evaluation"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────── compare ──
    def _compare(self, gt: Dict, pred: Dict) -> Dict[str, Any]:
        # ――― PO numbers ―――
        gt_po = {item.get("po_number", "") for item in gt.get("items", [])}
        pred_po = {item.get("po_number", "") for item in pred.get("items", [])}
        po_match = int(gt_po == pred_po)

        # ――― Line-item accuracy ―――
        gt_items = gt.get("items", [])
        pred_items = pred.get("items", [])
        total_items = len(gt_items)
        correct_items = 0
        for gt_item, pred_item in zip(gt_items, pred_items):
            qty_ok = gt_item.get("qty") == pred_item.get("qty")
            price_ok = _values_close(
                gt_item.get("unit_price", 0.0), pred_item.get("unit_price", 0.0), self.tol
            )
            total_ok = _values_close(
                gt_item.get("line_total", 0.0), pred_item.get("line_total", 0.0), self.tol
            )
            if qty_ok and price_ok and total_ok:
                correct_items += 1

        # ――― Totals ―――
        gt_totals = gt.get("totals", {})
        pred_totals = pred.get("totals", {})
        totals_match = int(
            all(
                _values_close(gt_totals.get(k, 0.0), pred_totals.get(k, 0.0), self.tol)
                for k in ("subtotal", "vat", "total")
                if k in gt_totals  # allow missing VAT in ground-truth
            )
        )

        return {
            "po_match": po_match,
            "po_gt": ",".join(sorted(gt_po)),
            "po_pred": ",".join(sorted(pred_po)),
            "line_items_correct": correct_items,
            "line_items_total": total_items,
            "line_item_accuracy": correct_items / total_items if total_items else 0.0,
            "totals_match": totals_match,
            "subtotal_gt": gt_totals.get("subtotal", 0.0),
            "subtotal_pred": pred_totals.get("subtotal", 0.0),
            "vat_gt": gt_totals.get("vat", 0.0),
            "vat_pred": pred_totals.get("vat", 0.0),
            "total_gt": gt_totals.get("total", 0.0),
            "total_pred": pred_totals.get("total", 0.0),
        }

    # ──────────────────────────────────────────────────────────────── public API ──
    def evaluate(self) -> None:
        """Populate *self.results* for every ground-truth file we find."""
        self.results.clear()
        for gt_path in sorted(self.gt_dir.glob("*.json")):
            name = gt_path.stem
            pred_path = self.pred_dir / name / "invoice.json"
            if not pred_path.exists():
                print(f"❌ Missing prediction for {name}")
                continue
            gt_json = _load_json(gt_path)
            pred_json = _load_json(pred_path)
            metrics = self._compare(gt_json, pred_json)
            metrics["file"] = name
            self.results.append(metrics)

    def _write_reports(self, df: pd.DataFrame, summary: Dict[str, float]) -> None:
        # Details
        details_csv = self.output_dir / "details.csv"
        details_json = self.output_dir / "details.json"
        df.to_csv(details_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        df.to_json(details_json, orient="records", indent=2)

        # Summary
        summary_json = self.output_dir / "summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"📄 Saved report to {self.output_dir.resolve()}")

    def report(self) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Return (summary, detailed_results) and save CSV/JSON on disk."""
        if not self.results:
            print("⚠️ No evaluation results – did you run evaluate()?")
            return {}, []

        df = pd.DataFrame(self.results, columns=self._CSV_COLUMNS)

        po_acc = df["po_match"].mean() * 100
        line_item_acc = (
            df["line_items_correct"].sum() / df["line_items_total"].sum() * 100
            if df["line_items_total"].sum() else 0.0
        )
        totals_acc = df["totals_match"].mean() * 100

        summary: Dict[str, float] = {
            "PO Accuracy (%)": round(po_acc, 2),
            "Line-item Accuracy (%)": round(line_item_acc, 2),
            "Total-fields Accuracy (%)": round(totals_acc, 2),
            "Num invoices": int(len(df)),
        }

        # Persist to disk
        self._write_reports(df, summary)

        # Console pretty-print
        print("\n=== 📊 Evaluation Summary ===")
        for k, v in summary.items():
            msg = f"{k:<25} {v:>6}" if k == "Num invoices" else f"{k:<25} {v:>6.2f}%"
            print(msg)

        return summary, self.results


# ────────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate invoice-extraction predictions.")
    parser.add_argument("--ground-truths", required=True, type=Path, help="Directory with ground-truth JSON files")
    parser.add_argument("--predictions", required=True, type=Path, help="Directory with model predictions (one sub-folder per PDF)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to save CSV/JSON reports (defaults to <predictions>/evaluation)")
    parser.add_argument("--tol", type=float, default=1e-2, help="Numeric tolerance when comparing floats (default 0.01)")
    args = parser.parse_args()

    evaluator = InvoiceEvaluator(args.ground_truths, args.predictions, args.out_dir)
    evaluator.evaluate()
    evaluator.report()