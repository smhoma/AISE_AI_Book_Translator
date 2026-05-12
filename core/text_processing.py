"""Text cleanup, paragraph merging, chunking, and Markdown formatting helpers."""

import html
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import nltk
except ImportError:
    nltk = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguageDirection:
    """Normalized language metadata used by Markdown, HTML, and PDF outputs."""

    lang: str
    direction: str


LANGUAGE_CODE_ALIASES = {
    "arabic": "ar",
    "ar": "ar",
    "ara": "ar",
    "العربية": "ar",
    "عربي": "ar",
    "فارسی": "fa",
    "فارسي": "fa",
    "پارسی": "fa",
    "پارسي": "fa",
    "farsi": "fa",
    "fa": "fa",
    "fas": "fa",
    "per": "fa",
    "persian": "fa",
    "he": "he",
    "heb": "he",
    "hebrew": "he",
    "iw": "he",
    "עברית": "he",
    "ur": "ur",
    "urd": "ur",
    "urdu": "ur",
    "اردو": "ur",
    "english": "en",
    "en": "en",
    "eng": "en",
    "french": "fr",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "spanish": "es",
    "es": "es",
    "spa": "es",
    "german": "de",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "italian": "it",
    "it": "it",
    "ita": "it",
    "portuguese": "pt",
    "pt": "pt",
    "por": "pt",
    "russian": "ru",
    "ru": "ru",
    "rus": "ru",
    "chinese": "zh",
    "zh": "zh",
    "zho": "zh",
    "japanese": "ja",
    "ja": "ja",
    "jpn": "ja",
    "korean": "ko",
    "ko": "ko",
    "kor": "ko",
    "turkish": "tr",
    "tr": "tr",
    "tur": "tr",
}

RTL_LANGUAGE_CODES = {"ar", "fa", "he", "ur"}
RTL_SCRIPT_PATTERN = re.compile(r"[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufefc]")


def _normalize_language_key(language: str | None) -> str:
    """Return a compact lookup key for human-readable language names."""
    if not language:
        return ""
    return re.sub(r"\s+", " ", language.strip().lower())


def get_language_code(language: str | None) -> str:
    """Return a best-effort BCP 47 language code for a human-readable language name."""
    normalized = _normalize_language_key(language)
    if not normalized:
        return "und"

    if normalized in LANGUAGE_CODE_ALIASES:
        return LANGUAGE_CODE_ALIASES[normalized]

    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", normalized):
        return normalized

    for token in re.split(r"[\s,/()_]+", normalized):
        if token in LANGUAGE_CODE_ALIASES:
            return LANGUAGE_CODE_ALIASES[token]

    return "und"


def get_text_direction(language: str | None) -> str:
    """Return ``rtl`` for known right-to-left target languages, otherwise ``ltr``."""
    lang_code = get_language_code(language).split("-", 1)[0]
    if lang_code in RTL_LANGUAGE_CODES:
        return "rtl"
    if lang_code == "und" and language and RTL_SCRIPT_PATTERN.search(language):
        return "rtl"
    return "ltr"


def get_language_direction(language: str | None) -> LanguageDirection:
    """Return normalized language code and text direction for a target language."""
    lang_code = get_language_code(language)
    return LanguageDirection(lang=lang_code, direction=get_text_direction(language))


def normalize_header_footer_text(text: str) -> str:
    """Normalize text for repeated header and footer comparisons."""
    normalized = text.strip().lower()[:80]
    normalized = re.sub(r"\d+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def detect_headers_and_footers(
    pages_paragraphs: List[List[Dict]],
    max_candidates_per_region: int = 2,
) -> Dict[str, set]:
    """Detect repeated header and footer paragraph patterns across pages."""
    logger.debug(
        "Detecting headers and footers. pages=%s max_candidates_per_region=%s",
        len(pages_paragraphs),
        max_candidates_per_region,
    )
    header_counter = Counter()
    footer_counter = Counter()
    num_pages = len(pages_paragraphs)

    for page_paras in pages_paragraphs:
        if not page_paras:
            continue

        for paragraph in page_paras[:max_candidates_per_region]:
            normalized = normalize_header_footer_text(paragraph["text"])
            if normalized:
                header_counter[normalized] += 1

        for paragraph in page_paras[-max_candidates_per_region:]:
            normalized = normalize_header_footer_text(paragraph["text"])
            if normalized:
                footer_counter[normalized] += 1

    header_threshold = max(2, int(0.3 * num_pages))
    footer_threshold = max(2, int(0.3 * num_pages))

    patterns = {
        "headers": {text for text, count in header_counter.items() if count >= header_threshold},
        "footers": {text for text, count in footer_counter.items() if count >= footer_threshold},
    }
    logger.debug("Header/footer detection complete. headers=%s footers=%s", len(patterns["headers"]), len(patterns["footers"]))
    return patterns


def remove_headers_and_footers(
    pages_paragraphs: List[List[Dict]],
    header_footer_patterns: Dict[str, set],
    max_candidates_per_region: int = 2,
) -> List[List[Dict]]:
    """Remove page-edge paragraphs that match detected header/footer patterns."""
    logger.debug(
        "Removing headers and footers. pages=%s header_patterns=%s footer_patterns=%s",
        len(pages_paragraphs),
        len(header_footer_patterns["headers"]),
        len(header_footer_patterns["footers"]),
    )
    headers = header_footer_patterns["headers"]
    footers = header_footer_patterns["footers"]
    cleaned_pages: List[List[Dict]] = []

    for page_paras in pages_paragraphs:
        if not page_paras:
            cleaned_pages.append([])
            continue

        new_page_paras = []
        for idx, paragraph in enumerate(page_paras):
            normalized = normalize_header_footer_text(paragraph["text"])

            if idx < max_candidates_per_region and normalized in headers:
                continue
            if idx >= len(page_paras) - max_candidates_per_region and normalized in footers:
                continue

            new_page_paras.append(paragraph)

        cleaned_pages.append(new_page_paras)

    before_count = sum(len(page) for page in pages_paragraphs)
    after_count = sum(len(page) for page in cleaned_pages)
    logger.debug("Header/footer removal complete. before=%s after=%s removed=%s", before_count, after_count, before_count - after_count)
    return cleaned_pages


def normalize_page_paragraphs(page_paragraphs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Annotate page paragraphs with indentation metadata and page statistics."""
    if not page_paragraphs:
        logger.debug("No page paragraphs to normalize")
        return [], {"baseline_left": 0}

    left_values = sorted(paragraph["left"] for paragraph in page_paragraphs)
    mid = len(left_values) // 2
    if len(left_values) % 2 == 1:
        baseline_left = left_values[mid]
    else:
        baseline_left = 0.5 * (left_values[mid - 1] + left_values[mid])

    indent_threshold = 20
    for idx, paragraph in enumerate(page_paragraphs):
        paragraph["indent"] = (paragraph["left"] - baseline_left) > indent_threshold
        paragraph["order_index"] = idx

    logger.debug("Normalized page paragraphs. count=%s baseline_left=%s", len(page_paragraphs), baseline_left)
    return page_paragraphs, {"baseline_left": baseline_left}


def ends_sentence(text: str) -> bool:
    """Return whether text ends with punctuation that normally closes a sentence."""
    return text.rstrip().endswith((".", "!", "?", ".”", "!”", "?”", ".'", "!'", "?'", ")"))


def looks_like_continuation(text: str) -> bool:
    """Return whether text appears to continue a sentence from a previous page."""
    stripped = text.strip()
    if not stripped:
        return False
    first_char = stripped[0]
    return first_char.islower() or first_char in "\"'“”‘’(-["


def likely_new_paragraph(paragraph: Dict) -> bool:
    """Return whether paragraph metadata suggests a new paragraph boundary."""
    text = paragraph["text"].lstrip()
    if not text:
        return False
    return bool(paragraph.get("indent") and text[0].isupper())


def merge_across_pages(pages_paragraphs: List[List[Dict]]) -> List[Dict]:
    """Merge paragraphs that appear to continue across page boundaries."""
    logger.debug("Merging paragraphs across pages. pages=%s", len(pages_paragraphs))
    merged: List[Dict] = []
    previous_paragraph: Dict | None = None

    for page_paras in pages_paragraphs:
        page_paras, _stats = normalize_page_paragraphs(page_paras)

        for idx, paragraph in enumerate(page_paras):
            if previous_paragraph is None:
                previous_paragraph = paragraph
                continue

            is_first_para_of_page = idx == 0
            if is_first_para_of_page:
                previous_text = previous_paragraph["text"]
                current_text = paragraph["text"]

                if likely_new_paragraph(paragraph):
                    merged.append(previous_paragraph)
                    previous_paragraph = paragraph
                    continue

                if (
                    not ends_sentence(previous_text)
                    and looks_like_continuation(current_text)
                    and not paragraph.get("indent")
                ):
                    previous_paragraph["text"] = previous_text.rstrip("-").rstrip() + " " + current_text.lstrip()
                    continue

            merged.append(previous_paragraph)
            previous_paragraph = paragraph

    if previous_paragraph is not None:
        merged.append(previous_paragraph)

    logger.debug("Merge across pages complete. merged_paragraphs=%s", len(merged))
    return merged


def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple character-count heuristic."""
    return max(1, len(text) // 4)


def _fallback_sentence_tokenize(paragraph: str) -> List[str]:
    """Split sentences with a small regex fallback when NLTK data is unavailable."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_paragraph_by_sentences(paragraph: str, max_tokens: int) -> List[str]:
    """Split an oversized paragraph into sentence-based chunks."""
    logger.debug("Splitting long paragraph by sentences. chars=%s max_tokens=%s", len(paragraph), max_tokens)
    if nltk is not None:
        try:
            sentences = nltk.sent_tokenize(paragraph)
        except LookupError:
            logger.debug("NLTK sentence tokenizer data is missing; using fallback tokenizer")
            sentences = _fallback_sentence_tokenize(paragraph)
    else:
        logger.debug("NLTK is not installed; using fallback tokenizer")
        sentences = _fallback_sentence_tokenize(paragraph)

    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    logger.debug("Sentence split complete. chunks=%s", len(chunks))
    return chunks


def chunk_paragraphs(paragraphs: List[str], max_tokens: int = 2500) -> List[str]:
    """Group paragraphs into translation chunks without splitting normal paragraphs."""
    logger.debug("Chunking paragraphs. paragraph_count=%s max_tokens=%s", len(paragraphs), max_tokens)
    chunks = []
    current_paras: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)

        if paragraph_tokens > max_tokens:
            sentence_chunks = split_paragraph_by_sentences(paragraph, max_tokens)
            for sentence_chunk in sentence_chunks:
                sentence_tokens = estimate_tokens(sentence_chunk)
                if current_tokens + sentence_tokens > max_tokens and current_paras:
                    chunks.append("\n\n".join(current_paras))
                    current_paras = []
                    current_tokens = 0
                current_paras.append(sentence_chunk)
                current_tokens += sentence_tokens
            continue

        if current_tokens + paragraph_tokens > max_tokens and current_paras:
            chunks.append("\n\n".join(current_paras))
            current_paras = []
            current_tokens = 0

        current_paras.append(paragraph)
        current_tokens += paragraph_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    logger.debug("Paragraph chunking complete. chunk_count=%s", len(chunks))
    return chunks


def looks_like_heading(text: str) -> bool:
    """Heuristically identify short all-caps translated headings."""
    stripped = text.strip()
    if len(stripped) == 0 or len(stripped) > 80:
        return False
    if stripped.endswith((".", "!", "?", ":", ";")):
        return False

    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(char.isupper() for char in letters) / len(letters)
    return upper_ratio > 0.6


def split_translated_paragraphs(translated_full: str) -> List[str]:
    """Split translated plain text into non-empty paragraphs."""
    text = translated_full.replace("\r\n", "\n")
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def _format_markdown_body(raw_paragraphs: List[str]) -> str:
    """Format translated paragraphs as Markdown without document-level metadata."""

    markdown_lines: List[str] = []
    for paragraph in raw_paragraphs:
        if looks_like_heading(paragraph):
            markdown_lines.append("# " + paragraph.strip())
        else:
            markdown_lines.append(paragraph.strip())
        markdown_lines.append("")

    return "\n".join(markdown_lines).strip() + "\n"


def format_translated_text_as_markdown(translated_full: str, target_lang: str | None = None) -> str:
    """Format translated plain text as Markdown while preserving paragraph breaks."""
    logger.debug("Formatting translated text as Markdown. chars=%s target_lang=%s", len(translated_full), target_lang)
    raw_paragraphs = split_translated_paragraphs(translated_full)
    markdown = _format_markdown_body(raw_paragraphs)
    language_direction = get_language_direction(target_lang)

    if language_direction.direction == "rtl":
        markdown = (
            f'<div dir="rtl" lang="{language_direction.lang}" align="right">\n\n'
            f"{markdown.rstrip()}\n\n"
            "</div>\n"
        )

    logger.debug("Markdown formatting complete. paragraphs=%s chars=%s", len(raw_paragraphs), len(markdown))
    return markdown


def _html_paragraph_content(paragraph: str) -> str:
    """Escape paragraph text for HTML and preserve intentional line breaks."""
    lines = [html.escape(line.strip()) for line in paragraph.split("\n")]
    return "<br>\n".join(lines)


def format_translated_text_as_html(
    translated_full: str,
    target_lang: str | None = None,
    title: str = "Translated Book",
) -> str:
    """Render translated plain text as a standalone HTML document."""
    logger.debug("Formatting translated text as HTML. chars=%s target_lang=%s", len(translated_full), target_lang)
    language_direction = get_language_direction(target_lang)
    text_align = "right" if language_direction.direction == "rtl" else "left"
    escaped_title = html.escape(title)
    raw_paragraphs = split_translated_paragraphs(translated_full)

    body_lines = []
    for paragraph in raw_paragraphs:
        content = _html_paragraph_content(paragraph)
        if looks_like_heading(paragraph):
            body_lines.append(f"<h1>{content}</h1>")
        else:
            body_lines.append(f"<p>{content}</p>")

    body = "\n".join(body_lines)
    document = f"""<!doctype html>
<html lang="{language_direction.lang}" dir="{language_direction.direction}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
    }}

    body {{
      margin: 0;
      background: #f7f7f5;
      color: #1f2328;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.8;
      direction: {language_direction.direction};
      text-align: {text_align};
    }}

    main {{
      box-sizing: border-box;
      width: min(100%, 78ch);
      margin: 0 auto;
      padding: 48px 28px;
      background: #ffffff;
      min-height: 100vh;
    }}

    h1 {{
      margin: 2.2rem 0 1rem;
      font-size: 1.8rem;
      line-height: 1.35;
    }}

    h1:first-child {{
      margin-top: 0;
    }}

    p {{
      margin: 0 0 1.25rem;
      font-size: 1.05rem;
    }}

    @media print {{
      body {{
        background: #ffffff;
      }}

      main {{
        width: auto;
        margin: 0;
        padding: 0;
        min-height: 0;
      }}
    }}
  </style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""
    logger.debug("HTML formatting complete. paragraphs=%s chars=%s", len(raw_paragraphs), len(document))
    return document
