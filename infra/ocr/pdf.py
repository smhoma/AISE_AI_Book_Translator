"""PDF rendering helpers for OCR preprocessing."""

import logging
from pathlib import Path
from typing import List

from core.exceptions import OCRProcessingError

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str | Path, dpi: int = 300, output_folder: str | Path = "pdf_pages") -> List[Path]:
    """Convert a PDF into ordered PNG page images for OCR."""
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    logger.info("Converting PDF to images. pdf_path=%s dpi=%s", pdf_path, dpi)

    if not pdf_path.exists():
        raise OCRProcessingError(f"PDF file does not exist: {pdf_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        from pdf2image import convert_from_path

        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as exc:
        logger.exception("PDF to image conversion failed for %s", pdf_path)
        raise OCRProcessingError(f"Failed to convert PDF to images: {exc}") from exc

    image_paths: List[Path] = []
    for idx, page in enumerate(pages, start=1):
        image_path = output_folder / f"page_{idx:04d}.png"
        page.save(image_path, "PNG")
        logger.debug("Saved PDF page image %s", image_path)
        image_paths.append(image_path)

    logger.info("PDF conversion complete. page_count=%s", len(image_paths))
    return image_paths
