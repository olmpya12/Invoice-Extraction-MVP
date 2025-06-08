import re
from typing import List, Dict, Any

# ----------------- Regex Patterns -------------------
PATTERNS = {
    "supplier_name": re.compile(
        r"^\s*(?P<supplier>[A-Z0-9 &\-\.,]{5,})\s*$", re.IGNORECASE
    ),
    "vat_number": re.compile(r"\bVAT[:\s]*([A-Z]{2}\d{8,12})\b", re.IGNORECASE),
    "invoice_no": re.compile(
        r"\bInvoice\s*Number[:\s]*(?P<inv>[A-Z0-9\-_]+)\b", re.IGNORECASE
    ),
    "invoice_date": re.compile(
        r"\b(?:Invoice\s*Date|Date)[:\s]*(?P<date>\d{2}[\/\-]\d{2}[\/\-]\d{4})\b",
        re.IGNORECASE,
    ),
    "po_number": re.compile(r"\bPO[-\s]*[:]?[\s]*(?P<po>\d{4,10})\b", re.IGNORECASE),
    "product_code": re.compile(
        r"Product\s*Code[:\s]*(?P<code>PRD[-A-Z0-9]+)", re.IGNORECASE
    ),
    "quantity": re.compile(r"\b(?:Quantity|Qty)[:\s]*(?P<qty>\d+)\b", re.IGNORECASE),
    "price": re.compile(
        r"\b(?:Unit\s*Price|Price)[:;\uFF1A]?\s*\$?(?P<pr>[0-9lI\.,]+)", re.IGNORECASE
    ),
    "line_total": re.compile(
        r"\b(?:Am[o0]unt|Total)[:;\uFF1A;\s]*\$?(?P<lt>[0-9lI\.,]+)\b", re.IGNORECASE
    ),
    "subtotal": re.compile(r"\bSubtotal[:\s]*\$?(?P<sub>[0-9\.,]+)\b", re.IGNORECASE),
    "vat_amount": re.compile(
        r"\bVAT\s*\(?\d{1,2}%\)?[:\s]*\$?(?P<vat>[0-9\.,]+)\b", re.IGNORECASE
    ),
    "total": re.compile(
        r"\bTotal\s*(?:A[mn][0o]unt|Amt)?[:\uFF1A]?\s*\$?(?P<tot>[0-9\.,]+)\b",
        re.IGNORECASE,
    ),
}


# ----------------- Normalization --------------------
def normalize_decimal(s: str) -> float:
    s = re.sub(r"(?<=\d)[lI](?=\d|\.)", "1", s)
    s = re.sub(r"(?<=\d)([lI])$", "1", s)
    s = re.sub(r"^([lI])(?=\d)", "1", s)
    s = re.sub(r"(?<=\d)O(?=\d|\.)", "0", s)
    s = re.sub(r"^O(?=[\d\.])", "0", s)
    s = re.sub(r"(?<=\d)O$", "0", s)
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


# ----------------- Field Extractors ------------------
def extract_supplier_info(text: str) -> Dict[str, str]:
    lines = text.splitlines()[:10]
    supplier = ""
    vat = ""
    for line in lines:
        if not supplier:
            m = PATTERNS["supplier_name"].match(line)
            if m and any(
                kw in line.lower() for kw in ["ltd", "inc", "corp", "solutions", "tech"]
            ):
                supplier = m.group("supplier").strip()
        if not vat:
            m = PATTERNS["vat_number"].search(line)
            if m:
                vat = m.group(1).strip()
        if supplier and vat:
            break
    return {"name": supplier, "vat": vat}


def extract_header_fields(text: str) -> Dict[str, str]:
    inv_no, date = "", ""
    m_inv = PATTERNS["invoice_no"].search(text)
    m_date = PATTERNS["invoice_date"].search(text)
    if m_inv:
        inv_no = m_inv.group("inv").strip()
    if m_date:
        date = m_date.group("date").strip()
    return {"invoice_no": inv_no, "date": date}


def extract_totals(text: str) -> Dict[str, float]:
    text = text.replace("Am0unt", "Amount")
    sub, vat, total = 0.0, None, 0.0
    m_sub = PATTERNS["subtotal"].search(text)
    m_vat = PATTERNS["vat_amount"].search(text)
    m_tot = PATTERNS["total"].search(text)
    if m_sub:
        sub = normalize_decimal(m_sub.group("sub"))
    if m_vat:
        vat = normalize_decimal(m_vat.group("vat"))
    if m_tot:
        total = normalize_decimal(m_tot.group("tot"))
    out = {"subtotal": sub}
    if vat is not None:
        out["vat"] = vat
    out["total"] = total
    return out


def extract_line_items(text: str, general_po: str) -> List[Dict[str, Any]]:
    """
    Extract items in three possible formats:
      1) Unnumbered “Hours: X x Rate: $Y” + “Amount: $Z”
      2) Numbered “1. …” or “9.Circuit Boards” style
      3) “Item Details:” style via line‐by‐line PRD‐code detection

    If an item has no per‐item PO, fallback to general_po (already prefixed “PO-…”).
    """
    items: List[Dict[str, Any]] = []

    # Normalize common OCR typos
    text = (
        text.replace("Am0unt", "Amount")
        .replace("H0urs", "Hours")
        .replace("×", "x")
        .replace("/hr", "")
    )

    # Split into non‐empty lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # --- Format #1: Unnumbered “Hours × Rate” items ---
    for i, line in enumerate(lines):
        m_hours = re.search(
            r"Hours[:\s]*(?P<hours>\d+)\s*x\s*Rate[:\s]*\$?(?P<rate>[0-9\.,]+)",
            line,
            re.IGNORECASE,
        )
        if m_hours:
            # Description is the previous non‐empty line, skipping header keywords
            desc = lines[i - 1] if i - 1 >= 0 else ""
            if desc.upper() in ["DESCRIPTION", "INVOICEDETAILS", "ITEM DETAILS:"]:
                desc = lines[i - 2] if i - 2 >= 0 else ""

            qty = int(m_hours.group("hours"))
            unit_price = normalize_decimal(m_hours.group("rate"))

            # Find “Amount” or “Total” within this line or next two lines
            line_total = 0.0
            for j in range(i, min(i + 3, len(lines))):
                m_lt = PATTERNS["line_total"].search(lines[j])
                if m_lt:
                    line_total = normalize_decimal(m_lt.group("lt"))
                    break

            items.append(
                {
                    "description": desc,
                    "product_code": "",
                    "qty": qty,
                    "unit_price": unit_price,
                    "line_total": line_total,
                    "po_number": general_po,
                }
            )

    # --- Format #2: Numbered blocks “X. …” (allow “9.” or “9. ”) ---
    blocks = re.split(r"\n(?=\d+\.)", text)
    for block in blocks:
        lines_blk = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines_blk or not re.match(r"^\d+\.\s*", lines_blk[0]):
            continue

        # Remove “X.” prefix (allow “X.” or “X. ”)
        first_line = re.sub(r"^\d+\.\s*", "", lines_blk[0]).strip()
        description = first_line
        code = ""
        qty = 0
        unit_price = 0.0
        line_total = 0.0
        po_number = general_po

        block_text = "\n".join(lines_blk[1:])

        m_code = PATTERNS["product_code"].search(block_text)
        if m_code:
            code = m_code.group("code").strip()

        m_qty = PATTERNS["quantity"].search(block_text)
        if m_qty:
            qty = int(m_qty.group("qty").strip())

        m_price = PATTERNS["price"].search(block_text)
        if m_price:
            unit_price = normalize_decimal(m_price.group("pr").strip())

        m_lt = PATTERNS["line_total"].search(block_text)
        if m_lt:
            line_total = normalize_decimal(m_lt.group("lt").strip())

        m_po = PATTERNS["po_number"].search(block_text)
        if m_po:
            po_number = f"PO-{m_po.group('po').strip()}"

        if code or line_total:
            items.append(
                {
                    "description": description,
                    "product_code": code,
                    "qty": qty,
                    "unit_price": unit_price,
                    "line_total": line_total,
                    "po_number": po_number,
                }
            )

    # ─── Format #3: “Item Details:” style via PRD code detection ────
    is_stop = False
    current_item: Dict[str, Any] = None
    for idx, line in enumerate(lines):
        # ── STOP if we hit the footer section (Subtotal, VAT or Total Amount) ──
        if is_stop:
            continue
        if PATTERNS["subtotal"].search(line):
            is_stop = True
            continue

        # If line is exactly “PRD-XXXX…”, start a new item
        if re.match(r"^PRD[-A-Z0-9]+$", line, re.IGNORECASE):
            # Save previous item if any
            if current_item is not None:
                items.append(current_item)

            current_item = {
                "description": "",
                "product_code": line,
                "qty": 0,
                "unit_price": 0.0,
                "line_total": 0.0,
                "po_number": general_po,
            }
            continue

        if current_item is None:
            continue

        if not current_item["description"]:
            current_item["description"] = line
            continue

        m_qty = PATTERNS["quantity"].search(line)
        if m_qty:
            current_item["qty"] = int(m_qty.group("qty"))
            continue

        m_price = PATTERNS["price"].search(line)
        if m_price:
            current_item["unit_price"] = normalize_decimal(m_price.group("pr"))
            continue
        m_lt = PATTERNS["line_total"].search(line)
        if m_lt:
            current_item["line_total"] = normalize_decimal(m_lt.group("lt"))
            continue

        m_po = PATTERNS["po_number"].search(line)
        if m_po:
            current_item["po_number"] = f"PO-{m_po.group('po')}"
            continue
        if re.match(r"^PO[:：]\s*(?:PO-?)?$", line, re.IGNORECASE) and (idx + 1) < len(
            lines
        ):
            next_line = lines[idx + 1].strip()
            m_next_full = re.match(r"^PO-?(\d{4,10})$", next_line, re.IGNORECASE)
            if m_next_full:
                current_item["po_number"] = f"PO-{m_next_full.group(1)}"
                continue
            m_next_digits = re.match(r"^(\d{4,10})$", next_line)
            if m_next_digits:
                current_item["po_number"] = f"PO-{m_next_digits.group(1)}"
                continue

    if current_item is not None:
        items.append(current_item)

    return items
