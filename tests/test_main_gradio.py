import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class MainGradioTests(unittest.TestCase):
    def test_translate_uploaded_pdf_returns_file_paths_as_strings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_pdf = temp_path / "sample.pdf"
            input_pdf.write_bytes(b"%PDF-1.4\n")

            output_path = temp_path / "translated_sample.md"
            html_output_path = temp_path / "translated_sample.html"
            pdf_output_path = temp_path / "translated_sample.pdf"
            for path in (output_path, html_output_path, pdf_output_path):
                path.write_text("translated", encoding="utf-8")

            fake_result = SimpleNamespace(
                output_path=output_path,
                html_output_path=html_output_path,
                pdf_output_path=pdf_output_path,
                page_count=1,
                paragraph_count=2,
                chunk_count=3,
            )

            with (
                patch("main._copy_pdf_to_input_dir", return_value=input_pdf),
                patch("main.BookTranslator") as translator_class,
            ):
                translator_class.return_value.translate_pdf.return_value = fake_result
                status, markdown_path, html_path, pdf_path = main.translate_uploaded_pdf(
                    str(input_pdf),
                    "English",
                    "Persian",
                )

            self.assertIn("Translation complete", status)
            self.assertEqual(markdown_path, str(output_path))
            self.assertEqual(html_path, str(html_output_path))
            self.assertEqual(pdf_path, str(pdf_output_path))


if __name__ == "__main__":
    unittest.main()
