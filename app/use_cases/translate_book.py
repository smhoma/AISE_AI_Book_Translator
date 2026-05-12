"""Application use case for translating OCR-readable PDF books."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from core.config import settings
from core.exceptions import FileProcessingError, OCRProcessingError
from core.output_rendering import render_html_to_pdf
from core.text_processing import (
    chunk_paragraphs,
    detect_headers_and_footers,
    estimate_tokens,
    format_translated_text_as_html,
    format_translated_text_as_markdown,
    get_text_direction,
    merge_across_pages,
    remove_headers_and_footers,
)
from app.use_cases.translate_chunks import OpenAIBookTranslator, load_translation_prompt_template
from infra.ocr.pdf import pdf_to_images
from infra.ocr.tesseract import process_page_to_paragraphs_from_tsv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationResult:
    """Artifacts and metrics produced by a completed PDF translation run."""

    output_path: Path
    raw_output_path: Path
    html_output_path: Path
    pdf_output_path: Path
    markdown_text: str
    html_text: str
    raw_text: str
    text_direction: str
    page_count: int
    paragraph_count: int
    chunk_count: int


class BookTranslator:
    """Coordinate PDF rendering, OCR, text cleanup, LLM translation, and output writing."""

    def __init__(
        self,
        llm_translator: OpenAIBookTranslator | None = None,
    ):
        """Create a translator use case with an optional custom LLM adapter."""
        self.llm_translator = llm_translator
        self.prompt_template = load_translation_prompt_template()
        logger.debug(
            "Initialized BookTranslator. custom_llm=%s prompt_template_loaded=%s",
            llm_translator is not None,
            self.prompt_template is not None,
        )

    def translate_pdf(
        self,
        pdf_path: str | Path,
        source_lang: str = settings.default_source_lang,
        target_lang: str = settings.default_target_lang,
        ocr_lang_code: str = settings.default_ocr_lang_code,
        max_tokens: int = settings.default_max_tokens,
        model_name: str | None = None,
        glossary: Optional[Dict[str, str]] = None,
    ) -> TranslationResult:
        """Translate a PDF and return generated output paths plus run metrics.

        Args:
            pdf_path: Path to the source PDF.
            source_lang: Human-readable source language name for the prompt.
            target_lang: Human-readable target language name for the prompt.
            ocr_lang_code: Tesseract language code used during OCR.
            max_tokens: Approximate maximum token budget for each translation chunk.
            model_name: Optional model override for the OpenAI-compatible client.
            glossary: Optional source-to-target glossary applied to every prompt.

        Returns:
            TranslationResult with output file paths, text content, and pipeline counts.
        """
        logger.info(
            "Starting PDF translation. pdf_path=%s source_lang=%s target_lang=%s ocr_lang_code=%s max_tokens=%s model_name=%s",
            pdf_path,
            source_lang,
            target_lang,
            ocr_lang_code,
            max_tokens,
            model_name,
        )
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileProcessingError(f"Input PDF does not exist: {pdf_path}")

        with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}_pages_") as temp_image_folder:
            logger.info("Converting PDF to page images: %s", pdf_path.name)
            image_paths = pdf_to_images(pdf_path, dpi=300, output_folder=temp_image_folder)
            if not image_paths:
                raise OCRProcessingError("PDF conversion did not produce any page images.")
            logger.info("PDF conversion produced %s page images", len(image_paths))

            pages_paragraphs = []
            for idx, image_path in enumerate(image_paths, start=1):
                logger.info("OCR and paragraph detection: page %s/%s", idx, len(image_paths))
                paragraphs = process_page_to_paragraphs_from_tsv(image_path, lang=ocr_lang_code)
                logger.debug("OCR extracted %s paragraphs from page %s", len(paragraphs), idx)
                pages_paragraphs.append(paragraphs)

        logger.info("Detecting and removing repeated headers and footers")
        header_footer_patterns = detect_headers_and_footers(pages_paragraphs)
        logger.debug(
            "Detected header/footer patterns. headers=%s footers=%s",
            len(header_footer_patterns["headers"]),
            len(header_footer_patterns["footers"]),
        )
        pages_paragraphs = remove_headers_and_footers(pages_paragraphs, header_footer_patterns)

        logger.info("Merging paragraphs across page boundaries")
        all_paragraphs = merge_across_pages(pages_paragraphs)
        if not all_paragraphs:
            raise OCRProcessingError("OCR did not extract any paragraphs from the PDF.")
        logger.info("Merged paragraph count: %s", len(all_paragraphs))

        paragraph_texts = [paragraph["text"] for paragraph in all_paragraphs]

        logger.info("Chunking text for translation")
        chunks = chunk_paragraphs(paragraph_texts, max_tokens=max_tokens)
        if not chunks:
            raise OCRProcessingError("No text chunks were created for translation.")
        logger.info("Created %s translation chunks", len(chunks))

        translator = self.llm_translator or OpenAIBookTranslator(
            model_name=model_name,
            prompt_template=self.prompt_template,
        )

        translated_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Translating chunk %s/%s (%s chars, approx %s tokens)",
                idx,
                len(chunks),
                len(chunk),
                estimate_tokens(chunk),
            )
            translated_chunks.append(
                translator.translate_chunk(
                    text=chunk,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=glossary,
                )
            )

        raw_text = "\n\n".join(translated_chunks)
        output_values = self.save_translation_outputs(raw_text, pdf_path, target_lang)

        return TranslationResult(
            **output_values,
            page_count=len(image_paths),
            paragraph_count=len(all_paragraphs),
            chunk_count=len(chunks),
        )

    @staticmethod
    def save_translation_outputs(raw_text: str, input_pdf_path: Path, target_lang: str) -> dict:
        """Save raw text, Markdown, HTML, and PDF outputs using the input filename stem."""
        settings.ensure_directories()
        output_name = f"translated_{input_pdf_path.stem}"
        raw_output_path = settings.output_dir / f"{output_name}.txt"
        output_path = settings.output_dir / f"{output_name}.md"
        html_output_path = settings.output_dir / f"{output_name}.html"
        pdf_output_path = settings.output_dir / f"{output_name}.pdf"

        logger.info("Saving translated outputs to %s", settings.output_dir)
        text_direction = get_text_direction(target_lang)
        markdown_text = format_translated_text_as_markdown(raw_text, target_lang=target_lang)
        html_text = format_translated_text_as_html(raw_text, target_lang=target_lang, title=input_pdf_path.stem)

        raw_output_path.write_text(raw_text, encoding="utf-8")
        output_path.write_text(markdown_text, encoding="utf-8")
        html_output_path.write_text(html_text, encoding="utf-8")
        logger.info("Saved translated text: %s", raw_output_path)
        logger.info("Saved translated Markdown: %s", output_path)
        logger.info("Saved translated HTML: %s", html_output_path)

        logger.info("Rendering translated PDF: %s", pdf_output_path)
        render_html_to_pdf(html_output_path, pdf_output_path)
        logger.info("Saved translated PDF: %s", pdf_output_path)

        return {
            "output_path": output_path,
            "raw_output_path": raw_output_path,
            "html_output_path": html_output_path,
            "pdf_output_path": pdf_output_path,
            "markdown_text": markdown_text,
            "html_text": html_text,
            "raw_text": raw_text,
            "text_direction": text_direction,
        }
