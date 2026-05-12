"""Tesseract OCR helpers that rebuild page text into paragraph records."""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from core.exceptions import OCRProcessingError

logger = logging.getLogger(__name__)


def ocr_page_tsv(image_path: str | Path, lang: str = "eng") -> List[Dict]:
    """Run Tesseract on one page image and return TSV rows as dictionaries."""
    logger.debug("Running Tesseract OCR. image_path=%s lang=%s", image_path, lang)
    try:
        from PIL import Image
        import pytesseract

        data = pytesseract.image_to_data(
            Image.open(image_path),
            lang=lang,
            config="--psm 1",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        logger.exception("Tesseract OCR failed for %s", image_path)
        raise OCRProcessingError(f"Failed to OCR page image {image_path}: {exc}") from exc

    rows = []
    for idx in range(len(data["level"])):
        rows.append({key: data[key][idx] for key in data.keys()})
    logger.debug("Tesseract OCR produced %s TSV rows for %s", len(rows), image_path)
    return rows


def build_paragraphs_from_tsv(tsv_rows: List[Dict]) -> List[Dict]:
    """Build paragraph records from word-level Tesseract TSV output."""
    logger.debug("Building paragraphs from TSV rows. row_count=%s", len(tsv_rows))
    para_lines: Dict[Tuple[int, int, int, int], List[Dict]] = defaultdict(list)

    for row in tsv_rows:
        if row["level"] != 5:
            continue
        if not row["text"] or row["text"].strip() == "":
            continue

        para_lines[
            (
                row["page_num"],
                row["block_num"],
                row["par_num"],
                row["line_num"],
            )
        ].append(row)

    paragraphs_meta: Dict[Tuple[int, int, int], Dict] = defaultdict(lambda: {"lines": [], "page_num": None})

    for (page_num, block_num, par_num, line_num), words in para_lines.items():
        words_sorted = sorted(words, key=lambda word: word["word_num"])
        line_words = [word["text"] for word in words_sorted if word["text"].strip()]
        if not line_words:
            continue

        line_text = re.sub(r"\s+", " ", " ".join(line_words)).strip()
        line_left = min(word["left"] for word in words_sorted)

        para_entry = paragraphs_meta[(page_num, block_num, par_num)]
        para_entry["page_num"] = page_num
        para_entry["lines"].append(
            {
                "line_num": line_num,
                "text": line_text,
                "left": line_left,
            }
        )

    logger.debug("Grouped TSV words into %s paragraph lines", len(para_lines))
    paragraph_dicts: List[Dict] = []

    for (page_num, block_num, par_num), para_entry in paragraphs_meta.items():
        lines_sorted = sorted(para_entry["lines"], key=lambda line: line["line_num"])
        fixed_lines: List[str] = []

        for idx, line_entry in enumerate(lines_sorted):
            line = line_entry["text"]
            if idx < len(lines_sorted) - 1:
                next_line = lines_sorted[idx + 1]["text"]
                current_match = re.search(r"(\w+)-\s*$", line)
                next_match = re.match(r"^(\w+)(.*)$", next_line)

                if current_match and next_match:
                    merged_word = current_match.group(1) + next_match.group(1)
                    rest_next = next_match.group(2)
                    line = re.sub(r"(\w+)-\s*$", merged_word, line)
                    lines_sorted[idx + 1]["text"] = rest_next.lstrip()

            fixed_lines.append(line)

        paragraph_text = re.sub(r"\s+", " ", " ".join(fixed_lines)).strip()
        if not paragraph_text:
            continue

        first_line_lefts = [line["left"] for line in lines_sorted[:1]]
        paragraph_left = sum(first_line_lefts) / len(first_line_lefts) if first_line_lefts else 0

        paragraph_dicts.append(
            {
                "text": paragraph_text,
                "left": paragraph_left,
                "indent_px": paragraph_left,
                "page_num": page_num,
                "block_num": block_num,
                "par_num": par_num,
            }
        )

    logger.debug("Built %s paragraph records from TSV", len(paragraph_dicts))
    return paragraph_dicts


def process_page_to_paragraphs_from_tsv(image_path: str | Path, lang: str = "eng") -> List[Dict]:
    """OCR one page image and return paragraph records in reading order."""
    rows = ocr_page_tsv(image_path, lang=lang)
    paragraphs = build_paragraphs_from_tsv(rows)
    logger.debug("Processed page to paragraphs. image_path=%s paragraphs=%s", image_path, len(paragraphs))
    return paragraphs
