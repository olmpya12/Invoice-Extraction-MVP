# Invoice Extraction MVP

This repository contains a **minimum‑viable prototype (MVP)** that automatically extracts structured data from invoice PDFs using OCR and machine‑learning techniques. It was developed as the submission for the interview task described in the accompanying brief.

---

## ✨ Key Points

* **Three independent extraction pipelines** are provided:

  1. **Regex Pipeline** – fast, interpretable rule‑based parsing.
  2. **LLM Pipeline** – by default uses a Groq‑hosted Llama‑70B model to convert raw invoice text to JSON; alternatively, you can point it to an OpenAI GPT‑4.1o‑mini endpoint and, with batch processing, inference costs work out to roughly **₺0.10 per invoice**.
  3. **Layout Pipeline (WIP)** – token‑classification with LayoutLMv3; requires fine‑tuning to reach production quality.
* **Common OCR Front‑End** – all pipelines share a high‑accuracy PaddleOCR pass (300 DPI, English) that converts each PDF page into text lines, confidences, and bounding boxes.
* **Modular CLI** – one entry‑point (`main.py`) with a `--method` flag; swap pipelines without touching the code.
* **Built‑in Evaluator** – `evaluation.py` compares predictions against ground‑truth JSON and produces CSV/JSON reports. *Ground‑truth labels were created manually and may not be 100 % accurate.*

---

## 📂 Repository Layout

```
.
├── src/
│   ├── InvoiceExtractors.py   # three pipeline classes
│   ├── OCRProcessor.py        # PaddleOCR wrapper
│   ├── Layout.py              # LayoutLMv3 helper
│   └── regex_extraction_helpers.py
├── main.py                    # unified CLI
├── evaluation.py              # metrics & reports
├── requirements.txt
└── ground_truths/             # gold‑standard JSON (3 sample invoices)
```

---

## 🔧 Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For GPU acceleration add:  paddlepaddle-gpu==2.6.1.post120  (CUDA 12.0 build)
```

> **Note** Set your Groq key before running the LLM pipeline:
>
> ```bash
> export GROQ_API_KEY="sk‑..."
> ```

---

## 🚀 Quick Start

Run any of the three pipelines on an invoice PDF:

```bash
# Regex
python main.py --method regex  --pdf samples/invoice1.pdf --out outputs

# LLM
python main.py --method llm    --pdf samples/invoice1.pdf --out outputs

# Layout (experimental)
python main.py --method layout --pdf samples/invoice1.pdf --out outputs
```

Outputs are written to `outputs/<method>/<invoice‑name>/`:

```
invoice.json        # structured prediction
ocr_stats.json      # OCR confidences
pages/              # rendered pages (.png)
texts/              # per‑page raw text
```

---

## 🧩 Pipeline Details

### 1. PaddleOCR Pre‑processing *(shared)*

Each extractor first calls `OCRProcessor` which:

1. Renders PDF pages at 300 DPI ➜ RGB images.
2. Runs PaddleOCR and returns:

   * **rec\_texts** – line texts
   * **rec\_scores** – confidence per line
   * **rec\_polys** – polygon boxes (converted to *xywh* for convenience)
3. Saves images/text and aggregates confidence statistics.

### 2. Regex Pipeline

`RegexInvoiceExtractor` applies hand‑crafted patterns (see `regex_extraction_helpers.py`) to the concatenated OCR text:

* Header fields – supplier, VAT, invoice #, date
* Line‑items – parses three common formats (hours × rate, numbered lists, PRD‑codes)
* Totals – subtotal / VAT / total

Achieved **100 % PO and line‑item accuracy** on the provided sample set (see `reports/`), **but this pipeline is heavily hard‑coded**. It reliably parses invoices that match the same template yet will struggle with unseen layouts; extending support to new suppliers requires ongoing pattern maintenance and incremental improvements.

### 3. LLM Pipeline

`LLMInvoiceExtractor` sends the OCR text (page‑delimited) to Groq’s `deepseek‑r1‑distill‑llama‑70b`, with a system prompt that forces it to **return valid JSON only**. Because large‑language models can reason over messy input—including typos, OCR artefacts, and unconventional layouts—this pipeline generalises to virtually any invoice template while still achieving perfect scores on the sample invoices.

However, the **current implementation performs only lightweight validation**: if the model returns malformed or empty JSON the extractor triggers a *single automatic retry*. In real‑world deployments you should add stronger schema guards, multi‑level fallbacks (e.g. secondary prompts, regex post‑patching), and business‑logic sanity checks on critical fields such as totals and PO‑to‑item consistency.

### 4. Layout Pipeline *(work in progress)*

`LayoutInvoiceExtractor` uses **LayoutLMv3** (`nielsr/layoutlmv3‑finetuned‑funsd` by default). The model consumes the **image** plus word‑level bounding boxes and predicts an entity label for each token (e.g. *B‑INVOICE\_NUMBER*, *B‑TOTAL*, *O*). Annotated pages are saved under `outputs/layout/` for inspection.

> Current off‑the‑shelf checkpoint is trained on FUNSD (forms) and therefore underperforms on invoices. Finetuning on an invoice‑specific dataset is required; until then this method is marked **incomplete** and excluded from accuracy reports.

---

## 📈 Evaluation

Compare predictions against ground truths:

```bash
python evaluation.py \
  --ground-truths ground_truths \
  --predictions   outputs/llm \
  --out-dir       reports/llm_eval
```

Example summary (`summary-llm.json`):

```json
{
  "PO Accuracy (%)": 100.0,
  "Line-item Accuracy (%)": 100.0,
  "Total-fields Accuracy (%)": 100.0,
  "Num invoices": 3
}
```

---

## 📝 Limitations & Future Work

* **Layout pipeline** – fine‑tune LayoutLMv3 on a curated invoice dataset and feed its token labels back into the JSON generator.
* **Hybrid strategy** – fall back to regex for simple layouts, LLM for complex / low‑confidence pages.
* **Batch OCR on GPU** – switch PaddleOCR to GPU and/or enable document unwarping & orientation for skewed scans.
* **Broader locale support** – add Turkish language pack and adapt regexes for multinational invoice formats.

---
