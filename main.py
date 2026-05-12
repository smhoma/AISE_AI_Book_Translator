"""Gradio entrypoint for the PDF book translator."""

import logging
import shutil
from pathlib import Path
from typing import Any

import gradio as gr

from app.use_cases.translate_book import BookTranslator
from core.config import settings
from core.logging_config import configure_logging

GRADIO_SERVER_PORT: int | None = None

logger = logging.getLogger(__name__)


def _uploaded_file_path(uploaded_file: Any) -> Path | None:
    """Return a local filesystem path from Gradio's uploaded file value."""
    if uploaded_file is None:
        return None

    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)

    file_name = getattr(uploaded_file, "name", None)
    if file_name:
        return Path(file_name)

    return None


def _copy_pdf_to_input_dir(uploaded_file: Any) -> Path:
    source_path = _uploaded_file_path(uploaded_file)
    if source_path is None:
        raise ValueError("Please upload a PDF file.")
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("Please upload a PDF file.")
    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded PDF does not exist: {source_path}")

    settings.ensure_directories()
    destination_path = settings.input_dir / source_path.name

    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
        logger.info("Copied uploaded PDF to %s", destination_path)
    else:
        logger.info("Uploaded PDF is already in the input directory: %s", destination_path)

    return destination_path


def translate_uploaded_pdf(uploaded_file: Any, source_lang: str, target_lang: str):
    """Translate an uploaded PDF and return status plus downloadable output files."""
    try:
        input_pdf_path = _copy_pdf_to_input_dir(uploaded_file)
        logger.info("Starting translation from Gradio for %s", input_pdf_path)

        result = BookTranslator().translate_pdf(
            pdf_path=input_pdf_path,
            source_lang=source_lang.strip() or settings.default_source_lang,
            target_lang=target_lang.strip() or settings.default_target_lang,
        )

        status = (
            f"Translation complete. Pages: {result.page_count}, "
            f"paragraphs: {result.paragraph_count}, chunks: {result.chunk_count}."
        )
        logger.info("Translation complete for %s", input_pdf_path)
        return status, str(result.output_path), str(result.html_output_path), str(result.pdf_output_path)

    except Exception as exc:
        logger.exception("Translation failed")
        return f"Translation failed: {exc}", None, None, None


def build_app() -> gr.Blocks:
    """Build the Gradio UI."""
    with gr.Blocks(title="Book Translator") as app:
        gr.Markdown("# Book Translator")

        pdf_input = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
        with gr.Row():
            source_lang_input = gr.Textbox(
                label="Source language",
                value=settings.default_source_lang,
            )
            target_lang_input = gr.Textbox(
                label="Target language",
                value=settings.default_target_lang,
            )

        translate_button = gr.Button("Translate", variant="primary")
        status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            markdown_output = gr.File(label="Markdown")
            html_output = gr.File(label="HTML")
            pdf_output = gr.File(label="PDF")

        translate_button.click(
            fn=translate_uploaded_pdf,
            inputs=[pdf_input, source_lang_input, target_lang_input],
            outputs=[status_output, markdown_output, html_output, pdf_output],
        )

    return app


def main() -> None:
    configure_logging()
    settings.ensure_directories()

    app = build_app()
    if GRADIO_SERVER_PORT is None:
        logger.info("Launching Gradio using the default port")
        app.launch()
    else:
        logger.info("Launching Gradio on hard-coded port %s", GRADIO_SERVER_PORT)
        app.launch(server_port=GRADIO_SERVER_PORT)


if __name__ == "__main__":
    main()
