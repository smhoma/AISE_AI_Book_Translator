import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.use_cases.translate_book import BookTranslator


class FakeSettings:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


class TranslationOutputTests(unittest.TestCase):
    def test_outputs_use_translated_prefix_and_input_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fake_settings = FakeSettings(output_dir)

            def fake_render_html_to_pdf(_html_path, pdf_path):
                Path(pdf_path).write_text("pdf", encoding="utf-8")
                return Path(pdf_path)

            with (
                patch("app.use_cases.translate_book.settings", fake_settings),
                patch("app.use_cases.translate_book.render_html_to_pdf", side_effect=fake_render_html_to_pdf),
            ):
                result = BookTranslator.save_translation_outputs(
                    raw_text="Translated text",
                    input_pdf_path=Path("my.book.pdf"),
                    target_lang="English",
                )

            self.assertEqual(result["raw_output_path"], output_dir / "translated_my.book.txt")
            self.assertEqual(result["output_path"], output_dir / "translated_my.book.md")
            self.assertEqual(result["html_output_path"], output_dir / "translated_my.book.html")
            self.assertEqual(result["pdf_output_path"], output_dir / "translated_my.book.pdf")

            for path in (
                result["raw_output_path"],
                result["output_path"],
                result["html_output_path"],
                result["pdf_output_path"],
            ):
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
