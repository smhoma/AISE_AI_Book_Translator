"""Helpers for rendering generated HTML outputs to PDF."""

import logging
import shutil
import subprocess
from pathlib import Path

from core.config import settings
from core.exceptions import FileProcessingError

logger = logging.getLogger(__name__)

DEFAULT_PDF_RENDERERS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)


def find_pdf_renderer(renderer_path: str | Path | None = None) -> str:
    """Find a Chrome-compatible executable for HTML-to-PDF rendering."""
    configured_renderer = renderer_path or settings.pdf_renderer_path

    if configured_renderer:
        configured_renderer = str(configured_renderer)
        expanded_path = Path(configured_renderer).expanduser()
        if expanded_path.exists():
            return str(expanded_path)

        renderer_on_path = shutil.which(configured_renderer)
        if renderer_on_path:
            return renderer_on_path

        raise FileProcessingError(f"PDF renderer not found: {configured_renderer}")

    for renderer_name in DEFAULT_PDF_RENDERERS:
        renderer_on_path = shutil.which(renderer_name)
        if renderer_on_path:
            return renderer_on_path

    renderer_names = ", ".join(DEFAULT_PDF_RENDERERS)
    raise FileProcessingError(
        "No PDF renderer found. Install Google Chrome/Chromium or set PDF_RENDERER_PATH. "
        f"Checked: {renderer_names}"
    )


def render_html_to_pdf(
    html_path: str | Path,
    pdf_path: str | Path,
    renderer_path: str | Path | None = None,
) -> Path:
    """Render an HTML file to PDF using headless Chrome/Chromium."""
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path).resolve()

    if not html_path.exists():
        raise FileProcessingError(f"HTML file does not exist: {html_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = find_pdf_renderer(renderer_path)
    command = [
        renderer,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]

    logger.info("Rendering HTML to PDF. renderer=%s html=%s pdf=%s", renderer, html_path, pdf_path)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise FileProcessingError(f"PDF rendering failed: {details}") from exc

    if not pdf_path.exists():
        raise FileProcessingError(f"PDF renderer completed without creating output: {pdf_path}")

    logger.info("HTML rendered to PDF: %s", pdf_path)
    return pdf_path
