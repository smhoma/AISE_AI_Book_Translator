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
