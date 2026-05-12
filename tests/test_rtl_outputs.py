import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.exceptions import FileProcessingError
from core.output_rendering import find_pdf_renderer, render_html_to_pdf
from core.text_processing import (
    format_translated_text_as_html,
    format_translated_text_as_markdown,
    get_language_code,
    get_text_direction,
)


class RTLOutputTests(unittest.TestCase):
    def test_detects_known_rtl_language_aliases(self):
        cases = {
            "Persian": "fa",
            "Farsi": "fa",
            "فارسی": "fa",
            "Arabic": "ar",
            "Hebrew": "he",
            "Urdu": "ur",
        }

        for language, expected_code in cases.items():
            with self.subTest(language=language):
                self.assertEqual(get_language_code(language), expected_code)
                self.assertEqual(get_text_direction(language), "rtl")

    def test_detects_ltr_language(self):
        self.assertEqual(get_language_code("English"), "en")
        self.assertEqual(get_text_direction("English"), "ltr")

    def test_ltr_markdown_matches_existing_format(self):
        translated_text = "TITLE\n\nFirst paragraph.\n\nSecond paragraph."
        expected = "# TITLE\n\nFirst paragraph.\n\nSecond paragraph.\n"

        self.assertEqual(format_translated_text_as_markdown(translated_text), expected)
        self.assertEqual(format_translated_text_as_markdown(translated_text, target_lang="English"), expected)

    def test_rtl_markdown_wraps_document_direction(self):
        translated_text = "عنوان\n\nاین یک پاراگراف است."
        markdown = format_translated_text_as_markdown(translated_text, target_lang="Persian")

        self.assertTrue(markdown.startswith('<div dir="rtl" lang="fa" align="right">\n\n'))
        self.assertIn("این یک پاراگراف است.", markdown)
        self.assertTrue(markdown.endswith("\n</div>\n"))

    def test_html_output_has_direction_and_escaped_text(self):
        html = format_translated_text_as_html(
            "عنوان\n\nمتن با <tag> و & علامت.",
            target_lang="فارسی",
            title="Book <Draft>",
        )

        self.assertIn('<html lang="fa" dir="rtl">', html)
        self.assertIn("<title>Book &lt;Draft&gt;</title>", html)
        self.assertIn("text-align: right;", html)
        self.assertIn("متن با &lt;tag&gt; و &amp; علامت.", html)


class PDFRenderingTests(unittest.TestCase):
    def test_render_html_to_pdf_calls_headless_chrome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            html_path = temp_path / "translated.html"
            pdf_path = temp_path / "translated.pdf"
            html_path.write_text("<html><body>ok</body></html>", encoding="utf-8")

            def fake_run(command, check, capture_output, text):
                self.assertTrue(check)
                self.assertTrue(capture_output)
                self.assertTrue(text)
                self.assertIn("--headless", command)
                self.assertIn("--print-to-pdf-no-header", command)
                self.assertIn(html_path.resolve().as_uri(), command)

                pdf_arg = next(arg for arg in command if arg.startswith("--print-to-pdf="))
                Path(pdf_arg.split("=", 1)[1]).write_bytes(b"%PDF-1.4")
                return subprocess.CompletedProcess(command, 0)

            with patch("core.output_rendering.find_pdf_renderer", return_value="/usr/bin/google-chrome"):
                with patch("core.output_rendering.subprocess.run", side_effect=fake_run) as run_mock:
                    result = render_html_to_pdf(html_path, pdf_path)

            self.assertEqual(result, pdf_path)
            self.assertTrue(pdf_path.exists())
            run_mock.assert_called_once()

    def test_missing_pdf_renderer_raises_clear_error(self):
        with patch("core.output_rendering.settings") as settings_mock:
            settings_mock.pdf_renderer_path = None
            with patch("core.output_rendering.shutil.which", return_value=None):
                with self.assertRaisesRegex(FileProcessingError, "No PDF renderer found"):
                    find_pdf_renderer()


if __name__ == "__main__":
    unittest.main()
