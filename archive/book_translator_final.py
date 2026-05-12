import os
import re
from typing import List, Tuple, Optional, Dict

from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import nltk
import openai

# Make sure tokenizer is available
# nltk.download('punkt')  # run once

# =====================================================
# CONFIG
# =====================================================

# Set your OpenAI API key (or rely on environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")  # or hardcode (not recommended)

# Choose model name
OPENAI_MODEL = "gpt-4.1-mini"  # change to the model you want


# =====================================================
# 1. PDF -> Page Images
# =====================================================

def pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    output_folder: str = "pdf_pages"
) -> List[str]:
    """
    Convert a PDF to page images using pdf2image.
    Returns a list of image file paths in order.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, page in enumerate(pages):
        img_name = f"page_{i+1:04d}.png"
        img_path = os.path.join(output_folder, img_name)
        page.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths


# =====================================================
# 2. Tesseract TSV-based OCR & Paragraph Extraction
# =====================================================

def ocr_page_tsv(image_path: str, lang: str = "eng") -> List[Dict]:
    """
    Run Tesseract on a single image page and return TSV rows as dicts.
    Each dict corresponds to a single word (or empty row for higher levels).
    """
    # `output_type=pytesseract.Output.DICT` returns a dictionary with lists
    data = pytesseract.image_to_data(
        Image.open(image_path),
        lang=lang,
        config="--psm 1",
        output_type=pytesseract.Output.DICT
    )
    rows = []
    n = len(data["level"])
    for i in range(n):
        row = {k: data[k][i] for k in data.keys()}
        rows.append(row)
    return rows


def build_paragraphs_from_tsv(tsv_rows: List[Dict]) -> List[str]:
    """
    Build paragraphs using Tesseract's block_num and par_num structure.
    For each (block_num, par_num), we merge all words in reading order.
    Then we merge lines in that paragraph and return para strings.
    """
    # Structure:
    # level: 1 - page, 2 - block, 3 - paragraph, 4 - line, 5 - word
    # We'll group by (block_num, par_num).
    from collections import defaultdict

    para_words = defaultdict(list)  # (block_num, par_num) -> list of (line_num, word_num, text)

    for row in tsv_rows:
        level = row["level"]
        text = row["text"]
        if level != 5:
            continue  # only word level
        if not text or text.strip() == "":
            continue

        block_num = row["block_num"]
        par_num = row["par_num"]
        line_num = row["line_num"]
        word_num = row["word_num"]

        para_words[(block_num, par_num)].append(
            (line_num, word_num, text)
        )

    # Now, for each paragraph group, sort by line_num, word_num, and join.
    paragraphs = []
    for (block_num, par_num), words in sorted(para_words.items(), key=lambda x: (x[0][0], x[0][1])):
        words_sorted = sorted(words, key=lambda w: (w[0], w[1]))
        # Optionally, you could recreate line breaks if you want,
        # but usually we want a continuous paragraph.
        para_text = " ".join(w[2] for w in words_sorted)
        para_text = clean_paragraph_text(para_text)
        if para_text:
            paragraphs.append(para_text)

    return paragraphs


def clean_paragraph_text(text: str) -> str:
    text = text.replace("\r\n", " ").replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_hyphenation_in_paragraph(paragraph: str) -> str:
    """
    Fix basic hyphenation if there are patterns like "para- graph".
    Because TSV is word-based, typical end-of-line hyphenation
    often appears as "para-" and "graph" separated by space.
    """
    # Replace "word- word" with "wordword".
    # This is heuristic; adjust or disable if it causes harm.
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", paragraph)


def process_page_to_paragraphs_from_tsv(image_path: str, lang: str = "eng") -> List[str]:
    rows = ocr_page_tsv(image_path, lang=lang)
    paragraphs = build_paragraphs_from_tsv(rows)
    paragraphs = [fix_hyphenation_in_paragraph(p) for p in paragraphs if p.strip()]
    return paragraphs


# =====================================================
# 3. Merge Paragraphs Across Pages
# =====================================================

def ends_sentence(text: str) -> bool:
    text = text.rstrip()
    return text.endswith((".", "!", "?", ".”", "!”", "?”", ".'", "!'", "?'", ")"))


def looks_like_continuation(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped:
        return False
    first_char = stripped[0]
    # If starts lower-case or is punctuation/quote that would appear mid-sentence
    if first_char.islower() or first_char in "\"'“”‘’(-[":
        return True
    # If it's a number or bullet, likely a new thing; return False
    return False


def merge_across_pages(pages_paragraphs: List[List[str]]) -> List[str]:
    """
    Merge logically continuous paragraphs across page boundaries:
    - If last paragraph of page N does not end a sentence and
      first paragraph of page N+1 looks like continuation -> merge.
    """
    merged: List[str] = []
    prev_para: Optional[str] = None

    for page_idx, page_paras in enumerate(pages_paragraphs):
        for i, para in enumerate(page_paras):
            if prev_para is None:
                prev_para = para
                continue

            is_first_para_of_page = (i == 0)
            if is_first_para_of_page and not ends_sentence(prev_para) and looks_like_continuation(para):
                prev_para = prev_para.rstrip("-").rstrip()
                prev_para = prev_para + " " + para.lstrip()
            else:
                merged.append(prev_para)
                prev_para = para

    if prev_para is not None:
        merged.append(prev_para)

    return merged


# =====================================================
# 4. Chunking for LLM (no broken paragraphs)
# =====================================================

def estimate_tokens(text: str) -> int:
    """
    Very rough token estimator: ~4 characters per token for Latin scripts.
    Replace with proper tokenizer if you want.
    """
    return max(1, len(text) // 4)


def split_paragraph_by_sentences(paragraph: str, max_tokens: int) -> List[str]:
    """
    Split a large paragraph into smaller fragments by sentence to fit within max_tokens.
    """
    sentences = nltk.sent_tokenize(paragraph)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        t = estimate_tokens(sent)
        if current_tokens + t > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_tokens = t
        else:
            current.append(sent)
            current_tokens += t

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def chunk_paragraphs(
    paragraphs: List[str],
    max_tokens: int = 3000,
    # overlap_paragraphs: int = 0
) -> List[str]:
    """
    Create translation chunks from paragraphs.
    - each chunk <= max_tokens (approx)
    - preserve paragraph boundaries
    - optional overlap of last N paragraphs between consecutive chunks
    """
    chunks = []
    current_paras: List[str] = []
    current_tokens = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        t = estimate_tokens(para)

        if t > max_tokens:
            # Split this long paragraph by sentences
            sentence_chunks = split_paragraph_by_sentences(para, max_tokens)
            for sc in sentence_chunks:
                st = estimate_tokens(sc)
                if current_tokens + st > max_tokens and current_paras:
                    chunks.append("\n\n".join(current_paras))
                    # if overlap_paragraphs > 0:
                        # Overlap not well-defined for sentence chunks;
                        # you can skip or treat each sentence-chunk as paragraph.
                        # current_paras = []
                    # else:
                        # current_paras = []
                    current_paras = []
                    current_tokens = 0
                current_paras.append(sc)
                current_tokens += st
            i += 1
            continue

        # If adding this paragraph would exceed the limit
        if current_tokens + t > max_tokens and current_paras:
            chunks.append("\n\n".join(current_paras))

            # if overlap_paragraphs > 0:
                # overlap = current_paras[-overlap_paragraphs:]
                # current_paras = overlap[:]
                # current_tokens = sum(estimate_tokens(p) for p in overlap)
            # else:
                # current_paras = []
                # current_tokens = 0
            current_paras = []
            current_tokens = 0

        current_paras.append(para)
        current_tokens += t
        i += 1

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


# =====================================================
# 5. OpenAI Translation
# =====================================================

def build_translation_prompt(
    text: str,
    glossary: Optional[Dict[str, str]],
    source_lang: str,
    target_lang: str
) -> str:
    glossary_str = ""
    if glossary:
        glossary_entries = "\n".join(f"{k} -> {v}" for k, v in glossary.items())
        glossary_str = (
            "\nHere is a glossary of terms and how they should be translated:\n"
            f"{glossary_entries}\n"
        )

    prompt = (
        f"You are translating a book from {source_lang} to {target_lang}.\n"
        "Instructions:\n"
        "- Translate the text accurately and naturally.\n"
        "- Do NOT summarize or omit any content.\n"
        "- Preserve paragraph breaks as in the original text.\n"
        "- Maintain the tone and style of the original.\n"
        "- Keep any inline markers (like **bold**, *italic*, etc.) untouched.\n"
        f"{glossary_str}\n"
        "Text to translate:\n"
        "-----\n"
        f"{text}\n"
        "-----\n"
        "Now output ONLY the translated text."
    )
    return prompt


def translate_chunk_with_openai(
    text: str,
    source_lang: str = "English",
    target_lang: str = "French",
    glossary: Optional[Dict[str, str]] = None,
    model: str = OPENAI_MODEL
) -> str:
    """
    Translate a chunk of text using OpenAI chat completion.
    """
    user_prompt = build_translation_prompt(text, glossary, source_lang, target_lang)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"You are a professional literary translator from {source_lang} to {target_lang}."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"].strip()


# =====================================================
# 6. Full Pipeline: PDF -> Translated Text
# =====================================================

def translate_pdf_book(
    pdf_path: str,
    source_lang: str = "English",
    target_lang: str = "French",
    ocr_lang_code: str = "eng",
    max_tokens: int = 3000,
    overlap_paragraphs: int = 1,
    temp_image_folder: str = "pdf_pages_tmp",
    glossary: Optional[Dict[str, str]] = None
) -> str:
    """
    Full pipeline:
    - PDF -> page images
    - Tesseract TSV OCR -> structured paragraphs (per page)
    - Merge paragraphs across pages
    - Chunk paragraphs for LLM
    - Translate each chunk via OpenAI
    - Return full translated text
    """
    # 1) PDF -> images
    print(f"Converting PDF to images: {pdf_path}")
    image_paths = pdf_to_images(pdf_path, dpi=300, output_folder=temp_image_folder)
    print(f"Total pages converted: {len(image_paths)}")

    # 2) OCR each page & build paragraphs via TSV
    pages_paragraphs: List[List[str]] = []
    for idx, img_path in enumerate(image_paths):
        print(f"OCR + paragraph detection for page {idx+1}/{len(image_paths)}: {img_path}")
        paras = process_page_to_paragraphs_from_tsv(img_path, lang=ocr_lang_code)
        pages_paragraphs.append(paras)

    # 3) Merge across pages
    print("Merging paragraphs across pages...")
    all_paragraphs = merge_across_pages(pages_paragraphs)
    print(f"Total paragraphs after merging: {len(all_paragraphs)}")

    # 4) Chunk
    print("Chunking paragraphs into LLM-sized chunks...")
    chunks = chunk_paragraphs(
        all_paragraphs,
        max_tokens=max_tokens,
        overlap_paragraphs=overlap_paragraphs
    )
    print(f"Total chunks for translation: {len(chunks)}")

    # 5) Translate each chunk
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)}...")
        translated = translate_chunk_with_openai(
            text=chunk,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary=glossary,
            model=OPENAI_MODEL
        )
        translated_chunks.append(translated)

    # 6) Merge translated chunks
    # If you used overlaps, you can add smarter de-duplication here.
    full_translation = "\n\n".join(translated_chunks)

    return full_translation


# =====================================================
# 7. Example main
# =====================================================

if __name__ == "__main__":
    # Example usage:
    input_pdf = "input_book.pdf"      # path to your PDF
    output_txt = "translated_book.txt"

    translated_text = translate_pdf_book(
        pdf_path=input_pdf,
        source_lang="English",        # adapt
        target_lang="Spanish",        # adapt
        ocr_lang_code="eng",          # Tesseract language code for source
        max_tokens=2500,
        overlap_paragraphs=1,
        temp_image_folder="book_pages_tmp",
        glossary=None                 # or pass a dict of term->translation
    )

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(translated_text)

    print(f"Done. Translated book saved to: {output_txt}")
import os
import re
import time
import random
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, Counter

from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import nltk
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, InternalServerError
from dotenv import load_dotenv

# import openai

load_dotenv()  # loads variables from .env

# Make sure tokenizer is available
# nltk.download('punkt')  # run once

# =====================================================
# CONFIG
# =====================================================

# Set your OpenAI API key (or rely on environment variable)
api_key = os.getenv("API_KEY")  # or hardcode (not recommended)
base_url = os.getenv("BASE_URL")  # or hardcode (not recommended)

client = OpenAI(api_key=api_key,
                base_url=base_url)


# Choose model name
OPENAI_MODEL = "gpt-4o"  # change to the model you want


# =====================================================
# 1. PDF -> Page Images
# =====================================================

def pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    output_folder: str = "pdf_pages"
) -> List[str]:
    """
    Convert a PDF to page images using pdf2image.
    Returns a list of image file paths in order.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, page in enumerate(pages):
        img_name = f"page_{i+1:04d}.png"
        img_path = os.path.join(output_folder, img_name)
        page.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths


# =====================================================
# 2. Tesseract TSV-based OCR & Paragraph Extraction
# =====================================================

def ocr_page_tsv(image_path: str, lang: str = "eng") -> List[Dict]:
    """
    Run Tesseract on a single image page and return TSV rows as dicts.
    Each dict corresponds to a single word (or empty row for higher levels).
    """
    # `output_type=pytesseract.Output.DICT` returns a dictionary with lists
    data = pytesseract.image_to_data(
        Image.open(image_path),
        lang=lang,
        config="--psm 1",
        output_type=pytesseract.Output.DICT
    )
    rows = []
    n = len(data["level"])
    for i in range(n):
        row = {k: data[k][i] for k in data.keys()}
        rows.append(row)
    return rows


def build_paragraphs_from_tsv(tsv_rows: List[Dict]) -> List[Dict]:
    """
    Build paragraphs using Tesseract's block_num and par_num structure,
    but *line aware* so we can safely fix end-of-line hyphenation.

    Returns a list of paragraph dicts:
      {
        "text": "...",
        "left": <avg left of first line>,
        "indent_px": <indent for first line in px>,
        "page_num": int,
      }
    """
    # para_lines[(page, block, par, line)] -> list of word dicts
    para_lines: Dict[Tuple[int, int, int, int], List[Dict]] = defaultdict(list)

    for row in tsv_rows:
        level = row["level"]
        text = row["text"]
        if level != 5:
            continue
        if not text or text.strip() == "":
            continue

        page_num = row["page_num"]
        block_num = row["block_num"]
        par_num = row["par_num"]
        line_num = row["line_num"]

        para_lines[(page_num, block_num, par_num, line_num)].append(row)

    # Now within each paragraph, we have multiple lines. We'll:
    # 1) sort lines by line_num
    # 2) sort words by word_num
    # 3) fix end-of-line hyphenation *between lines*
    # 4) join into a single paragraph string
    paragraphs_meta: Dict[Tuple[int, int, int], Dict] = defaultdict(lambda: {
        "lines": [],
        "page_num": None,
    })

    # Collect lines per paragraph key (page, block, par)
    for (page_num, block_num, par_num, line_num), words in para_lines.items():
        words_sorted = sorted(words, key=lambda w: w["word_num"])
        line_texts = [w["text"] for w in words_sorted if w["text"].strip()]
        if not line_texts:
            continue

        line_text = " ".join(line_texts)
        line_text = re.sub(r"\s+", " ", line_text).strip()

        # Geometry: we can approximate line-left as min(left) over its words
        line_left = min(w["left"] for w in words_sorted)

        para_key = (page_num, block_num, par_num)
        para_entry = paragraphs_meta[para_key]
        para_entry["page_num"] = page_num
        para_entry["lines"].append({
            "line_num": line_num,
            "text": line_text,
            "left": line_left,
        })

    paragraph_dicts: List[Dict] = []

    for (page_num, block_num, par_num), para_entry in paragraphs_meta.items():
        # Sort lines by line_num
        lines_sorted = sorted(para_entry["lines"], key=lambda ln: ln["line_num"])

        # Fix hyphenation across line boundaries *within this paragraph*
        fixed_lines: List[str] = []
        i = 0
        while i < len(lines_sorted):
            line = lines_sorted[i]["text"]
            if i < len(lines_sorted) - 1:
                next_line = lines_sorted[i+1]["text"]

                # If line ends with "<word>-" and next_line starts with "<word>"
                # we merge them: "para-" + "graph" -> "paragraph".
                m = re.search(r"(\w+)-\s*$", line)
                n = re.match(r"^(\w+)(.*)$", next_line)
                if m and n:
                    first_part = m.group(1)
                    second_part = n.group(1)

                    # Merge: "para-" + "graph" -> "paragraph"
                    merged_word = first_part + second_part
                    rest_next = n.group(2)  # remainder of next_line after that word
                    line = re.sub(r"(\w+)-\s*$", merged_word, line)
                    next_line = rest_next.lstrip()
                    lines_sorted[i+1]["text"] = next_line

            fixed_lines.append(line)
            i += 1

        # Now join fixed_lines into a paragraph
        para_text = " ".join(fixed_lines)
        para_text = re.sub(r"\s+", " ", para_text).strip()
        if not para_text:
            continue

        # Approx paragraph left: average left of first line (or min)
        first_line_lefts = [ln["left"] for ln in lines_sorted[:1]]
        para_left = sum(first_line_lefts) / len(first_line_lefts) if first_line_lefts else 0

        paragraph_dicts.append({
            "text": para_text,
            "left": para_left,
            "indent_px": para_left,  # relative indent; we'll normalize per page later
            "page_num": page_num,
            "block_num": block_num,
            "par_num": par_num,
        })

    return paragraph_dicts


def clean_paragraph_text(text: str) -> str:
    text = text.replace("\r\n", " ").replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_page_to_paragraphs_from_tsv(image_path: str, lang: str = "eng") -> List[Dict]:
    rows = ocr_page_tsv(image_path, lang=lang)
    paragraphs = build_paragraphs_from_tsv(rows)
    return paragraphs


# =====================================================
# 3. Header/Footer detection
# =====================================================

def normalize_header_footer_text(text: str) -> str:
    """
    Normalize text for header/footer comparison:
      - lowercase
      - strip spaces
      - remove digits (to ignore page numbers)
    """
    t = text.strip().lower()
    # Optionally shorten: keep only first ~80 characters to avoid overfitting large paragraphs
    t = t[:80]
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def detect_headers_and_footers(pages_paragraphs: List[List[Dict]], max_candidates_per_region: int = 2) -> Dict[str, set]:
    """
    Identify repeated header/footer lines across pages.
    Returns a dict:
      {
        "headers": set of normalized strings,
        "footers": set of normalized strings,
      }
    """
    header_counter = Counter()
    footer_counter = Counter()
    num_pages = len(pages_paragraphs)

    for page_paras in pages_paragraphs:
        if not page_paras:
            continue
        # First N and last N candidates
        header_candidates = page_paras[:max_candidates_per_region]
        footer_candidates = page_paras[-max_candidates_per_region:]

        for p in header_candidates:
            norm = normalize_header_footer_text(p["text"])
            if norm:
                header_counter[norm] += 1

        for p in footer_candidates:
            norm = normalize_header_footer_text(p["text"])
            if norm:
                footer_counter[norm] += 1

    # Threshold: appears on at least 30% of pages
    header_threshold = max(2, int(0.3 * num_pages))
    footer_threshold = max(2, int(0.3 * num_pages))

    headers = {t for t, c in header_counter.items() if c >= header_threshold}
    footers = {t for t, c in footer_counter.items() if c >= footer_threshold}

    return {"headers": headers, "footers": footers}


def remove_headers_and_footers(
    pages_paragraphs: List[List[Dict]],
    hf_patterns: Dict[str, set],
    max_candidates_per_region: int = 2
) -> List[List[Dict]]:
    """
    Remove paragraphs matching detected headers/footers from each page.
    """
    headers = hf_patterns["headers"]
    footers = hf_patterns["footers"]

    cleaned_pages: List[List[Dict]] = []

    for page_paras in pages_paragraphs:
        if not page_paras:
            cleaned_pages.append([])
            continue

        new_page_paras = []
        for idx, p in enumerate(page_paras):
            norm = normalize_header_footer_text(p["text"])

            # Only aggressively apply header patterns to early paragraphs
            if idx < max_candidates_per_region and norm in headers:
                continue
            # Likewise for footer patterns
            if idx >= len(page_paras) - max_candidates_per_region and norm in footers:
                continue

            new_page_paras.append(p)

        cleaned_pages.append(new_page_paras)

    return cleaned_pages


# =====================================================
# 4. Merge Paragraphs Across Pages
# =====================================================

def normalize_page_paragraphs(
    page_paragraphs: List[Dict],
    header_footer_threshold: float = 0.2
) -> Tuple[List[Dict], Dict]:
    """
    Given all paragraphs (dicts) on a *single page*, compute:
      - baseline left margin (mode/median),
      - indent flag.
    Also returns some page stats.

    For now just returns paragraphs with 'indent' flag and 'top_bucket'
    approximated via order in list.
    """

    # Here we only have text-level info and 'left'; no vertical coordinate per paragraph
    # in this version. We can approximate vertical ordering based on input order.
    # For header/footer detection across pages we need only text patterns, which we'll do later.
    if not page_paragraphs:
        return [], {"baseline_left": 0}

    # Baseline left as median of all paragraph lefts (robust to outliers)
    left_values = sorted(p["left"] for p in page_paragraphs)
    mid = len(left_values) // 2
    if len(left_values) % 2 == 1:
        baseline_left = left_values[mid]
    else:
        baseline_left = 0.5 * (left_values[mid - 1] + left_values[mid])

    # Mark indent if paragraph left is significantly greater than baseline
    # threshold is arbitrary: you might tweak based on DPI and font size.
    indent_threshold = 20  # pixels; adjust as needed

    for idx, p in enumerate(page_paragraphs):
        p["indent"] = (p["left"] - baseline_left) > indent_threshold
        p["order_index"] = idx  # approximate vertical order

    return page_paragraphs, {"baseline_left": baseline_left}


def ends_sentence(text: str) -> bool:
    text = text.rstrip()
    return text.endswith((".", "!", "?", ".”", "!”", "?”", ".'", "!'", "?'", ")"))


def looks_like_continuation(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    first_char = s[0]
    # Lowercase start is a strong continuation signal
    if first_char.islower():
        return True
    # If it starts with a quote or bracket, could be continuation of dialogue/thought
    if first_char in "\"'“”‘’(-[":
        return True
    return False


def likely_new_paragraph(p: Dict) -> bool:
    """
    Heuristic: clearly a new paragraph if:
      - indented, and
      - starts with uppercase (not obviously mid-sentence).
    You could enrich this with other heuristics.
    """
    text = p["text"].lstrip()
    if not text:
        return False
    first_char = text[0]
    # Indentation is a strong sign
    if p.get("indent") and first_char.isupper():
        return True
    return False


def merge_across_pages(
    pages_paragraphs: List[List[Dict]]
) -> List[Dict]:
    """
    Merge logically continuous paragraphs across page boundaries, using:
      - punctuation (ends_sentence),
      - casing (looks_like_continuation),
      - indentation metadata (indent).
    Also assumes each sublist is a page in reading order.
    """
    merged: List[Dict] = []
    prev_para: Dict = None

    for page_idx, page_paras in enumerate(pages_paragraphs):
        # Normalize indentation per page
        page_paras, _stats = normalize_page_paragraphs(page_paras)

        for i, para in enumerate(page_paras):
            if prev_para is None:
                prev_para = para
                continue

            is_first_para_of_page = (i == 0)

            if is_first_para_of_page:
                prev_text = prev_para["text"]
                curr_text = para["text"]

                # Strong "definitely new paragraph" condition
                if likely_new_paragraph(para):
                    merged.append(prev_para)
                    prev_para = para
                    continue

                # Candidate for continuation
                if (not ends_sentence(prev_text)) and looks_like_continuation(curr_text) and not para.get("indent"):
                    # Merge texts
                    merged_text = prev_text.rstrip("-").rstrip() + " " + curr_text.lstrip()
                    prev_para["text"] = merged_text
                    # we keep prev_para's metadata (page_num, etc.) from first part
                    continue
                else:
                    merged.append(prev_para)
                    prev_para = para
            else:
                # Not first paragraph of page: do NOT auto-merge across paragraphs on same page,
                # we assume Tesseract got par boundaries reasonably well.
                merged.append(prev_para)
                prev_para = para

    if prev_para is not None:
        merged.append(prev_para)

    return merged


# =====================================================
# 5. Chunking for LLM (no broken paragraphs)
# =====================================================

def estimate_tokens(text: str) -> int:
    """
    Very rough token estimator: ~4 characters per token for Latin scripts.
    Replace with proper tokenizer if you want.
    """
    return max(1, len(text) // 4)


def split_paragraph_by_sentences(paragraph: str, max_tokens: int) -> List[str]:
    """
    Split a large paragraph into smaller fragments by sentence to fit within max_tokens.
    """
    sentences = nltk.sent_tokenize(paragraph)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        t = estimate_tokens(sent)
        if current_tokens + t > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_tokens = t
        else:
            current.append(sent)
            current_tokens += t

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def chunk_paragraphs(
    paragraphs: List[str],
    max_tokens: int = 2500,
    # overlap_paragraphs: int = 0
) -> List[str]:
    """
    Create translation chunks from paragraphs.
    - each chunk <= max_tokens (approx)
    - preserve paragraph boundaries
    - optional overlap of last N paragraphs between consecutive chunks
    """
    chunks = []
    current_paras: List[str] = []
    current_tokens = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        t = estimate_tokens(para)

        if t > max_tokens:
            # Split this long paragraph by sentences
            sentence_chunks = split_paragraph_by_sentences(para, max_tokens)
            for sc in sentence_chunks:
                st = estimate_tokens(sc)
                if current_tokens + st > max_tokens and current_paras:
                    chunks.append("\n\n".join(current_paras))
                    # if overlap_paragraphs > 0:
                        # Overlap not well-defined for sentence chunks;
                        # you can skip or treat each sentence-chunk as paragraph.
                        # current_paras = []
                    # else:
                        # current_paras = []
                    current_paras = []
                    current_tokens = 0
                current_paras.append(sc)
                current_tokens += st
            i += 1
            continue

        # If adding this paragraph would exceed the limit
        if current_tokens + t > max_tokens and current_paras:
            chunks.append("\n\n".join(current_paras))

            # if overlap_paragraphs > 0:
                # overlap = current_paras[-overlap_paragraphs:]
                # current_paras = overlap[:]
                # current_tokens = sum(estimate_tokens(p) for p in overlap)
            # else:
                # current_paras = []
                # current_tokens = 0
            current_paras = []
            current_tokens = 0

        current_paras.append(para)
        current_tokens += t
        i += 1

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


# =====================================================
# 6. OpenAI Translation
# =====================================================

def build_translation_prompt(
    text: str,
    glossary: Optional[Dict[str, str]],
    source_lang: str,
    target_lang: str
) -> str:
    glossary_str = ""
    if glossary:
        glossary_entries = "\n".join(f"{k} -> {v}" for k, v in glossary.items())
        glossary_str = (
            "\nHere is a glossary of terms and how they should be translated:\n"
            f"{glossary_entries}\n"
        )

    prompt = (
        f"You are a professional book translator. You are translating a book from {source_lang} to {target_lang}.\n"
        "Instructions:\n"
        "- Translate the text accurately and naturally.\n"
        "- Do NOT summarize or omit any content.\n"
        "- Do not add commentary.\n"
        "- DO NOT translate parts of text that does not need translation like codes snippets, formulas, etc.\n"
        "- Preserve paragraph breaks as in the original text.\n"
        "- Maintain the tone and style and the structure of the original.\n"
        "- Keep any inline markers (like **bold**, *italic*, etc.) untouched.\n"
        f"{glossary_str}\n"
        "Text to translate:\n"
        "-----\n"
        f"{text}\n"
        "-----\n"
        "Now output ONLY the translated text."
    )
    return prompt


def call_with_retries(
    func,
    *args,
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True,
    **kwargs
):
    """
    Call an OpenAI client function with exponential backoff retries on transient errors.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except (APITimeoutError, RateLimitError, APIError, InternalServerError) as e:
            # For APIError, only retry on 5xx
            status = getattr(e, "status", None)
            if isinstance(e, APIError) and (status is not None and status < 500):
                # 4xx APIError – not retriable
                raise

            if attempt == max_retries - 1:
                # Last attempt, re-raise
                raise

            # Compute backoff delay
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random() / 2.0)  # 0.5–1x

            print(f"[WARN] OpenAI transient error ({type(e).__name__}: {e}). "
                  f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)


def translate_chunk_with_openai(
    text: str,
    source_lang: str = "English",
    target_lang: str = "فارسی",
    glossary: Optional[Dict[str, str]] = None,
    model: str = OPENAI_MODEL
) -> str:
    """
    Translate a chunk of text using OpenAI chat completion.
    """
    user_prompt = build_translation_prompt(text, glossary, source_lang, target_lang)

    response = call_with_retries(
        client.chat.completions.create,
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"You are a professional literary translator from {source_lang} to {target_lang}."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=8000,        # to avoid default value and truncation of the output
    )

    if response.choices[0].finish_reason == "length":
        print(f"[WARNING] Translation truncated due to max_tokens limit!")

    return response.choices[0].message.content.strip()


# =====================================================
# 7. Markdown Output Format
# =====================================================

def looks_like_heading(text: str) -> bool:
    """
    Very simple heuristic:
      - short line (<= 80 chars),
      - no ending punctuation like '.' or '?',
      - high ratio of uppercase letters.
    """
    t = text.strip()
    if len(t) == 0 or len(t) > 80:
        return False
    if t.endswith((".", "!", "?", ":", ";")):
        return False

    # uppercase ratio
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio > 0.6


def format_translated_text_as_markdown(translated_full: str) -> str:
    """
    Convert the translated text into reasonably nice Markdown:
      - Split paragraphs by blank lines.
      - Detect headings and prepend '# '.
      - Ensure a blank line between paragraphs.
    """
    # Normalize line endings
    text = translated_full.replace("\r\n", "\n")

    # Split into paragraphs on double newlines
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    md_lines: List[str] = []

    for para in raw_paragraphs:
        if looks_like_heading(para):
            md_lines.append("# " + para.strip())
        else:
            # Normal paragraph
            md_lines.append(para.strip())
        md_lines.append("")  # blank line after each block

    # Join with newlines
    markdown = "\n".join(md_lines).strip() + "\n"
    return markdown

# =====================================================
# 8. Full Pipeline: PDF -> Translated Text
# =====================================================

def translate_pdf_book(
    pdf_path: str,
    source_lang: str = "English",
    target_lang: str = "فارسی",
    ocr_lang_code: str = "eng",
    max_tokens: int = 2500,
    # overlap_paragraphs: int = 1,
    temp_image_folder: str = "pdf_pages_tmp",
    glossary: Optional[Dict[str, str]] = None
) -> str:
    # 1) PDF -> images
    print(f"Converting PDF to images: {pdf_path}")
    image_paths = pdf_to_images(pdf_path, dpi=300, output_folder=temp_image_folder)
    print(f"Total pages converted: {len(image_paths)}")

    # 2) OCR each page & build paragraphs via TSV (with metadata)
    pages_paragraphs: List[List[Dict]] = []
    for idx, img_path in enumerate(image_paths):
        print(f"OCR + paragraph detection for page {idx+1}/{len(image_paths)}: {img_path}")
        paras = process_page_to_paragraphs_from_tsv(img_path, lang=ocr_lang_code)
        pages_paragraphs.append(paras)

    print(f"[DEBUG] Last page has {len(pages_paragraphs[-1])} paragraphs")
    if pages_paragraphs[-1]:
        print(f"[DEBUG] Last paragraph on last page: {pages_paragraphs[-1][-1]['text'][:100]}")    

    # 3) Detect and remove repeated headers/footers
    print("Detecting and removing headers/footers...")
    hf_patterns = detect_headers_and_footers(pages_paragraphs)
    pages_paragraphs = remove_headers_and_footers(pages_paragraphs, hf_patterns)

    print(f"[DEBUG] After H/F removal, last page has {len(pages_paragraphs[-1])} paragraphs")

    # 4) Merge across pages
    print("Merging paragraphs across pages...")
    all_paragraphs = merge_across_pages(pages_paragraphs)
    print(f"Total paragraphs after merging: {len(all_paragraphs)}")

    print(f"[DEBUG] Total merged paragraphs: {len(all_paragraphs)}")
    print(f"[DEBUG] Last merged paragraph: {all_paragraphs[-1]['text'][:100]}")

    # 5) Extract just text for chunking
    paragraph_texts = [p["text"] for p in all_paragraphs]

    # 6) Chunk
    print("Chunking paragraphs into LLM-sized chunks...")
    chunks = chunk_paragraphs(
        paragraph_texts,
        max_tokens=max_tokens,
        # overlap_paragraphs=overlap_paragraphs
    )
    print(f"Total chunks for translation: {len(chunks)}")
    
    print("=== CHUNK SUMMARY ===")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)}, chars={len(c)}, tokens~={estimate_tokens(c)}")
        print(c[:200].replace("\n", " ") + "...")
        print("------")
  
    # 7) Translate each chunk
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"\n[INFO] Translating chunk {i+1}/{len(chunks)}")
        print("[DEBUG] Chunk preview:", chunk[-200:].replace("\n", " "), "...")
        translated = translate_chunk_with_openai(
            text=chunk,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary=glossary,
            model=OPENAI_MODEL
        )
        print("[DEBUG] Translated preview:", translated[-200:].replace("\n", " "), "...")
        translated_chunks.append(translated)

    # 8) Merge translated chunks
    translated_full = "\n\n".join(translated_chunks)

    # 9) Post-process to Markdown
    markdown = format_translated_text_as_markdown(translated_full)
    # return markdown

    with open("raw_translated.txt", "w", encoding="utf-8") as f:
        f.write(translated_full)
    return translated_full




# =====================================================
# 9. Example main
# =====================================================

if __name__ == "__main__":
    input_pdf = "How_To_Remember_Everything_You_Read.pdf"
    output_md = f"translated_{input_pdf}.md"

    markdown_text = translate_pdf_book(
        pdf_path=os.path.join("session06_project1", input_pdf),
        source_lang="English",
        target_lang="فارسی",
        ocr_lang_code="eng",
        max_tokens=2500,
        # overlap_paragraphs=0,
        temp_image_folder="book_pages_tmp",
        glossary=None
    )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Done. Translated book (Markdown) saved to: {output_md}")
