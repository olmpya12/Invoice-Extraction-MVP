#!/usr/bin/env python3
"""
Unified invoice extraction.

Usage:
  python main.py --method regex  --pdf path/to/invoice.pdf
  python main.py --method llm    --pdf path/to/invoice.pdf
  python main.py --method layout --pdf path/to/invoice.pdf
"""

import argparse
from pathlib import Path

from src.InvoiceExtractors import (
    RegexInvoiceExtractor,
    LLMInvoiceExtractor,
    LayoutInvoiceExtractor,
)
from openai import OpenAI


def run_regex_pipeline(pdf_path: str, output_dir: str):
    extractor = RegexInvoiceExtractor(Path(pdf_path), Path(output_dir))
    extractor.extract()


def run_llm_pipeline(pdf_path: str, output_dir: str):
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key="put-api-key",  # replace with your key or use env-var
    )
    model = "deepseek-r1-distill-llama-70b"
    system_prompt = """You are an invoice-extraction engine.
                    Return ONLY valid JSON with this schema:
                    {
                    "supplier": { "name": str, "vat": str },
                    "invoice_no": str,
                    "date": str,
                    "items": [ {
                        "description": str, "product_code": str,
                        "qty": int, "unit_price": float, "line_total": float, "po_number": str } ],
                    "totals":   "totals": {
                                            "subtotal": float,
                                            "vat": float,
                                            "total": float
                                            }}"""
    extractor = LLMInvoiceExtractor(
        Path(pdf_path), Path(output_dir), client, model, system_prompt
    )
    extractor.extract()


def run_layout_pipeline(pdf_path: str, output_dir: str):
    extractor = LayoutInvoiceExtractor(Path(pdf_path), Path(output_dir))
    extractor.extract()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoice Extraction Runner")
    parser.add_argument(
        "--method",
        required=True,
        choices=["regex", "llm", "layout"],
        help="Extraction method",
    )
    parser.add_argument(
        "--pdf", required=True, type=str, help="Path to the invoice PDF"
    )
    parser.add_argument("--out", default="outputs", type=str, help="Output directory")

    args = parser.parse_args()
    pdf_path   = Path(args.pdf).expanduser().resolve()
    file_stem  = pdf_path.stem 
    
    output_dir = Path(args.out) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.pdf).exists():
        raise FileNotFoundError(f"PDF file not found: {args.pdf}")

    method_dispatch = {
        "regex": run_regex_pipeline,
        "llm": run_llm_pipeline,
        "layout": run_layout_pipeline,
    }

    method_dispatch[args.method](args.pdf, output_dir)

