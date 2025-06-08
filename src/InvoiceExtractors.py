import time
import json
from pathlib import Path
import statistics
from src.OCRProcessor import OCRProcessor
from typing import List, Dict, Any
from src.Layout import LayoutLvm3
import re


class BaseInvoiceExtractor:
    def __init__(self, pdf_path: Path, output_dir: Path):
        self.pdf_path = pdf_path

        pdf_name = pdf_path.stem
        self.output_dir = output_dir / pdf_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ocr_processor = OCRProcessor()
        self.pages_text: List[str] = []
        self.all_scores: List[float] = []
        self.stats: Dict[str, Any] = {}
        self.layout_data: List[Dict[str, Any]] = []

    # ── OCR phase (saves text files, images, layout_input.json) ──────
    def save_ocr_results(self):
        base_out = self.output_dir
        pages_dir = base_out / "pages"
        texts_dir = base_out / "texts"
        pages_dir.mkdir(parents=True, exist_ok=True)
        texts_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(
            self.ocr_processor.pdf_to_images(self.pdf_path), start=1
        ):
            # ----- Save resized image
            method = str(self.output_dir).split("/")[-2]
            if method == "layout":
                img = img.resize((762, 1000))

            img.save(pages_dir / f"page{idx}.png")
            # ----- OCR
            text, scores, boxes = self.ocr_processor.run_ocr(img)
            (texts_dir / f"page{idx}.txt").write_text(text, encoding="utf8")

            # ----- Stats per page
            mean_conf = round(statistics.fmean(scores) * 100, 2) if scores else 0.0
            stdev_conf = (
                round(statistics.stdev(scores) * 100, 2) if len(scores) > 1 else 0.0
            )
            self.stats[f"page_{idx}"] = {
                "mean_conf": mean_conf,
                "stdev_conf": stdev_conf,
            }

            self.pages_text.append(text)
            self.all_scores.extend(scores)

            # ----- Layout info per line
            for b, t, s in zip(boxes, text.split("\n"), scores):
                self.layout_data.append(
                    {
                        "page": idx,
                        "text": t,
                        "score": round(s, 3),
                        "box": b,
                    }
                )

            print(f"✓ OCR Page {idx}: {mean_conf}% mean confidence")

        # ----- Global stats
        if self.all_scores:
            self.stats["overall_mean_conf"] = round(
                statistics.fmean(self.all_scores) * 100, 2
            )
            self.stats["overall_stdev_conf"] = round(
                statistics.stdev(self.all_scores) * 100, 2
            )
        else:
            self.stats["overall_mean_conf"] = self.stats["overall_stdev_conf"] = 0.0

        (base_out / "ocr_stats.json").write_text(json.dumps(self.stats, indent=2))


class RegexInvoiceExtractor(BaseInvoiceExtractor):
    def __init__(self, pdf_path: Path, output_dir: Path):
        super().__init__(pdf_path, output_dir)

    def extract(self):
        start_time = time.time()
        self.save_ocr_results()
        combined_text = "\n".join(self.pages_text)

        from src.regex_extraction_helpers import (
            extract_supplier_info,
            extract_header_fields,
            extract_line_items,
            extract_totals,
        )

        m_global_po = re.search(
            r"\bPONUMBER[:\s]*PO[-\s]*(?P<po>\d{4,10})\b", combined_text, re.IGNORECASE
        )
        general_po = f"PO-{m_global_po.group('po')}" if m_global_po else ""

        first_page_text = self.pages_text[0] if self.pages_text else ""
        supplier_info = extract_supplier_info(first_page_text)
        header_fields = extract_header_fields(combined_text)
        items = extract_line_items(combined_text, general_po)
        totals = extract_totals(combined_text)

        result = {
            "supplier": supplier_info,
            "invoice_no": header_fields.get("invoice_no", ""),
            "date": header_fields.get("date", ""),
            "items": items,
            "totals": totals,
        }
        (self.output_dir / "invoice.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        print(f"🏁 Extraction complete in {round(time.time() - start_time, 2)}s")


class LLMInvoiceExtractor(BaseInvoiceExtractor):
    def __init__(
        self, pdf_path: Path, output_dir: Path, llm_client, model: str, sys_prompt: str
    ):
        super().__init__(pdf_path, output_dir)
        self.client = llm_client
        self.model = model
        self.sys_prompt = sys_prompt

    def extract(self):
        start_time = time.time()
        self.save_ocr_results()
        combined_text = "\n".join(
            f"=== Page {i+1} ===\n{text}" for i, text in enumerate(self.pages_text)
        )

        def call_llm():
            return self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": combined_text},
                ],
            )

        response = call_llm()
        usage = response.usage.model_dump() if response.usage else {}

        # Re-ask if response is suspiciously short
        if usage and usage.get("completion_tokens", 0) < usage.get("prompt_tokens", 0):
            print("⚠️ Output tokens shorter than input — retrying once...")
            response = call_llm()
            usage = response.usage.model_dump() if response.usage else {}

        result = json.loads(response.choices[0].message.content)
        (self.output_dir / "invoice.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )

        usage.update(
            {"model": self.model, "elapsed_sec": round(time.time() - start_time, 2)}
        )
        (self.output_dir / "usage.json").write_text(json.dumps(usage, indent=2))

        print(
            f"🏁 Done in {usage['elapsed_sec']}s | prompt={usage.get('prompt_tokens','?')}, completion={usage.get('completion_tokens','?')} tokens"
        )


class LayoutInvoiceExtractor(BaseInvoiceExtractor):
    def __init__(
        self,
        pdf_path: Path,
        output_dir: Path,
        model_name="nielsr/layoutlmv3-finetuned-funsd",
    ):
        super().__init__(pdf_path, output_dir)
        self.layout_model = LayoutLvm3(model_name=model_name)

    def extract(self):
        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for page_idx, img in enumerate(
            self.ocr_processor.pdf_to_images(self.pdf_path), start=1
        ):
            img = img.resize((762, 1000))
            text, scores, boxes = self.ocr_processor.run_ocr(img)
            lines = text.split("\n")

            predictions, processed_boxes = self.layout_model.infer(img, lines, boxes)

            annotated = self.layout_model.draw(
                img.copy(), lines, processed_boxes, predictions
            )
            annotated.save(self.output_dir / f"page{page_idx}_layout.png")

            print(f"✓ Page {page_idx}: Layout processed")

        elapsed = round(time.time() - start_time, 2)
        print(f"🏁 Layout-based extraction complete in {elapsed}s")
