from paddleocr import PaddleOCR
from PIL import Image
import fitz
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import re


class OCRProcessor:
    def __init__(self, dpi=300, lang="en"):
        self.dpi = dpi
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        doc = fitz.open(pdf_path)
        return [
            Image.frombytes(
                "RGB",
                (
                    page.get_pixmap(dpi=self.dpi).width,
                    page.get_pixmap(dpi=self.dpi).height,
                ),
                page.get_pixmap(dpi=self.dpi).samples,
            )
            for page in doc
        ]

    def run_ocr(
        self, img: Image.Image
    ) -> Tuple[str, List[float], List[List[List[int]]]]:
        """Run OCR and get line‑level layout.

        Returns:
            text   – concatenated line texts separated by newlines
            scores – list of confidences per line
            boxes  – list of 4‑point polygons [[x,y],...] per line (PaddleOCR order)
        """
        result = self.ocr.predict(np.array(img))[0]  # (boxes, (text, score)) per line

        texts = result.get("rec_texts", [])
        scores = [float(s) for s in result.get("rec_scores", [])]

        polys = result.get("rec_polys")

        boxes = []
        for poly in polys:
            x_coords = [pt[0] for pt in poly]
            y_coords = [pt[1] for pt in poly]
            x, y = min(x_coords), min(y_coords)
            w, h = max(x_coords) - x, max(y_coords) - y
            boxes.append([x, y, w, h])

        clean_text = "\n".join(texts).strip()
        clean_text = re.sub(r"\s{2,}", " ", clean_text)
        return clean_text, scores, boxes
